/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2017 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_expr_nlhdlr_perspective.c
 * @brief  perspective nonlinear handler
 * @author Ksenia Bestuzheva
 */

#include <string.h>

#include "scip/cons_varbound.h"
#include "scip/cons_expr_nlhdlr_perspective.h"
#include "scip/cons_expr.h"
#include "scip/cons_expr_var.h"
#include "scip/cons_expr_sum.h"
#include "scip/scip_sol.h"
#include "struct_cons_expr.h"

/* fundamental nonlinear handler properties */
#define NLHDLR_NAME         "perspective"
#define NLHDLR_DESC         "perspective handler for expressions"
#define NLHDLR_PRIORITY     75

#define DETECTSUM    FALSE
#define MULTCUTS     TRUE

/*
 * Data structures
 */

/** data structure to store information of a semicontinuous variable */
struct SCIP_SCVarData
{
   SCIP_Real*            vals0;              /**< values of the variable when the corresponding bvars[i] = 0 */
   SCIP_VAR**            bvars;              /**< the binary variables on which the variable domain depends */
   int                   nbnds;              /**< number of suitable on/off bounds the var has */
   int                   bndssize;           /**< size of the arrays */
};
typedef struct SCIP_SCVarData SCIP_SCVARDATA;

/** nonlinear handler expression data */
struct SCIP_ConsExpr_NlhdlrExprData
{
   SCIP_EXPRCURV         curvature;          /**< curvature of the expression */

   SCIP_CONSEXPR_EXPR**  onoffterms;         /**< on/off terms for which we apply perspective cuts */
   SCIP_Real*            onoffcoefs;         /**< coefficients of onoffterms */
   SCIP_VAR***           termbvars;          /**< binary vars associated with onoffterms */
   int*                  ntermbvars;         /**< number of binary variables for each term */
   int                   nonoffterms;        /**< number of on/off expressions */
   int                   onofftermssize;     /**< size of arrays describing on/off terms */

   SCIP_CONSEXPR_EXPR**  convterms;          /**< convex terms for which we apply gradient cuts */
   SCIP_Real*            convcoefs;          /**< coefficients of convterms */
   int                   nconvterms;         /**< number of convterms */
   int                   convtermssize;      /**< size of the convterms array */

   SCIP_CONSEXPR_EXPR**  varexprs;           /**< variable expressions */
   int                   nvarexprs;          /**< total number of variable expressions */
};

/** nonlinear handler data */
struct SCIP_ConsExpr_NlhdlrData
{
   SCIP_HASHMAP*         scvars;             /**< maps semicontinuous variables to their on/off bounds */
};

/*
 * Local methods
 */

/** checks if a variable is semicontinuous and, if needed, updates the hashmap
 *
 * A variable is semicontinuous if its bounds depend on the binary variable bvar and bvar == 0 => var = v_off for some
 * real constant v_off. If the bvar is not specified, find the first binary variable that var depends on.
 */
static
SCIP_RETCODE varIsSemicontinuous(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< the variable to check */
   SCIP_HASHMAP*         scvars,             /**< semicontinuous variable information */
   SCIP_Bool*            result              /**< buffer to store whether var is semicontinuous */
   )
{
   SCIP_Real lb0, ub0, lb1, ub1, glb, gub;
   SCIP_Bool exists;
   int c, pos, newsize;
   SCIP_VAR** vlbvars;
   SCIP_VAR** vubvars;
   SCIP_Real* vlbcoefs;
   SCIP_Real* vubcoefs;
   SCIP_Real* vlbconstants;
   SCIP_Real* vubconstants;
   int nvlbs, nvubs;
   SCIP_SCVARDATA* scvdata;
   SCIP_VAR* bvar;

   assert(scip != NULL);
   assert(var != NULL);
   assert(scvars != NULL);
   assert(result != NULL);

   *result = FALSE;

   scvdata = (SCIP_SCVARDATA*) SCIPhashmapGetImage(scvars, (void*)var);
   if( scvdata != NULL )
   {
      *result = TRUE;
      return SCIP_OKAY;
   }

   vlbvars = SCIPvarGetVlbVars(var);
   vubvars = SCIPvarGetVubVars(var);
   vlbcoefs = SCIPvarGetVlbCoefs(var);
   vubcoefs = SCIPvarGetVubCoefs(var);
   vlbconstants = SCIPvarGetVlbConstants(var);
   vubconstants = SCIPvarGetVubConstants(var);
   nvlbs = SCIPvarGetNVlbs(var);
   nvubs = SCIPvarGetNVubs(var);
   glb = SCIPvarGetLbGlobal(var);
   gub = SCIPvarGetUbGlobal(var);

   /* Scan through lower bounds; for each binary vlbvar save the corresponding lb0 and lb1.
    * Then check if there is an upper bound with this vlbvar and save ub0 and ub1.
    * If the found bounds imply that the var value is fixed to some val0 when vlbvar = 0,
    * save vlbvar and val0 to scvdata.
    */
   for( c = 0; c < nvlbs; ++c )
   {
      SCIPdebugMsg(scip, "\nvar %s lower bound: lbvar = %s, coef = %f, const = %f", SCIPvarGetName(var), SCIPvarGetName(vlbvars[c]), vlbcoefs[c], vlbconstants[c]);

      if( SCIPvarGetType(vlbvars[c]) != SCIP_VARTYPE_BINARY )
         continue;

      bvar = vlbvars[c];

      lb0 = MAX(vlbconstants[c], glb);
      lb1 = MAX(vlbconstants[c] + vlbcoefs[c], glb);

      /* look for bvar in vubvars */
      if( vubvars != NULL )
         exists = SCIPsortedvecFindPtr((void**)vubvars, SCIPvarComp, bvar, nvubs, &pos);
      else
         exists = FALSE;
      if( exists )
      {/*lint --e{644}*/
         /* save the upper bounds */
         ub0 = MIN(vubconstants[pos], gub);
         ub1 = MIN(vubconstants[pos] + vubcoefs[pos], gub);
      }
      else
      {
         /* if there is no upper bound with vubvar = bvar, use global var bounds */
         ub0 = gub;
         ub1 = gub;
      }

      /* the 'off' domain of a semicontinuous var should reduce to a single point and be different from the 'on' domain */
      SCIPdebugMsg(scip, "\nbnds for this var are: %f, %f, %f, %f", lb0, lb1, ub0, ub1);
      if( lb0 == ub0 && (lb0 != lb1 || ub0 != ub1) ) /*lint !e777*/
      {
         if( scvdata == NULL )
         {
            SCIP_CALL( SCIPallocClearBlockMemory(scip, &scvdata) );
         }

         if( scvdata->nbnds + 1 > scvdata->bndssize )
         {
            newsize = SCIPcalcMemGrowSize(scip, scvdata->nbnds + 1);
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->bvars,  scvdata->bndssize, newsize) );
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->vals0, scvdata->bndssize, newsize) );
            scvdata->bndssize = newsize;
         }
         assert(scvdata->nbnds + 1 <= scvdata->bndssize);

         scvdata->bvars[scvdata->nbnds] = bvar;
         scvdata->vals0[scvdata->nbnds] = lb0;
         ++scvdata->nbnds;
      }
   }

   /* look for vubvars that have not been processed yet */
   assert(vubvars != NULL || nvubs == 0);
   for( c = 0; c < nvubs; ++c )
   {
      SCIPdebugMsg(scip, "\nvar %s upper bound: ubvar = %s, coef = %f, const = %f",
         SCIPvarGetName(var), SCIPvarGetName(vubvars[c]), vubcoefs[c], vubconstants[c]);  /*lint !e613*/

      if( SCIPvarGetType(vubvars[c]) != SCIP_VARTYPE_BINARY)  /*lint !e613*/
         continue;

      bvar = vubvars[c];  /*lint !e613*/

      /* skip vars that are in vlbvars */
      if( vlbvars != NULL && SCIPsortedvecFindPtr((void**)vlbvars, SCIPvarComp, bvar, nvlbs, &pos) )
         continue;

      lb0 = glb;
      lb1 = glb;
      ub0 = MIN(vubconstants[c], gub);
      ub1 = MIN(vubconstants[c] + vubcoefs[c], gub);

      /* the 'off' domain of a semicontinuous var should reduce to a single point and be different from the 'on' domain */
      if( lb0 == ub0 && (lb0 != lb1 || ub0 != ub1) ) /*lint !e777*/
      {
         if( scvdata == NULL )
         {
            SCIP_CALL( SCIPallocClearBlockMemory(scip, &scvdata) );
         }

         if( scvdata->nbnds + 1 > scvdata->bndssize )
         {
            newsize = SCIPcalcMemGrowSize(scip, scvdata->nbnds + 1);
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->bvars, scvdata->bndssize, newsize) );
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->vals0, scvdata->bndssize, newsize) );
            scvdata->bndssize = newsize;
         }
         assert(scvdata->nbnds + 1 <= scvdata->bndssize);

         scvdata->bvars[scvdata->nbnds] = bvar;
         scvdata->vals0[scvdata->nbnds] = lb0;
         ++scvdata->nbnds;
      }
   }

   SCIPdebugMsg(scip, "\nvar %s has bounds %f, %f", SCIPvarGetName(var), glb, gub);

   if( scvdata != NULL )
   {
      /* sort bvars and vals0 */
      SCIPsortPtrReal((void**)scvdata->bvars, scvdata->vals0, SCIPvarComp, scvdata->nbnds);
      SCIPdebugMsg(scip, " and the following on/off bounds:");
      for( c = 0; c < scvdata->nbnds; ++c )
      {
         SCIPdebugMsg(scip, "\nc = %d, bvar %s: val0 = %f", c, SCIPvarGetName(scvdata->bvars[c]), scvdata->vals0[c]);
      }
      SCIP_CALL( SCIPhashmapInsert(scvars, var, scvdata) );
      *result = TRUE;
   }

   return SCIP_OKAY;
}

/** adds an expression to the array of on/off expressions */
static
SCIP_RETCODE addOnoffTerm(
   SCIP*                         scip,                     /**< SCIP data structure */
   SCIP_CONSHDLR*                conshdlr,                 /**< constraint handler */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata,           /**< expression data */
   SCIP_Real                     coef,                     /**< coef of the added term */
   SCIP_CONSEXPR_EXPR*           expr,                     /**< expr to add */
   SCIP_VAR**                    bvars,                    /**< binary variables */
   int                           nbvars                    /**< number of binary variables */
)
{
   int newsize;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(expr != NULL);
   assert(bvars != NULL);

   if( nlhdlrexprdata->nonoffterms + 1 > nlhdlrexprdata->onofftermssize )
   {
      newsize = SCIPcalcMemGrowSize(scip, nlhdlrexprdata->nonoffterms + 1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->onoffterms,  nlhdlrexprdata->onofftermssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->onoffcoefs, nlhdlrexprdata->onofftermssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->termbvars, nlhdlrexprdata->onofftermssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->ntermbvars, nlhdlrexprdata->onofftermssize, newsize) );
      nlhdlrexprdata->onofftermssize = newsize;
   }
   assert(nlhdlrexprdata->nonoffterms + 1 <= nlhdlrexprdata->onofftermssize);

   nlhdlrexprdata->onoffcoefs[nlhdlrexprdata->nonoffterms] = coef;
   nlhdlrexprdata->onoffterms[nlhdlrexprdata->nonoffterms] = expr;
   nlhdlrexprdata->termbvars[nlhdlrexprdata->nonoffterms] = bvars;
   nlhdlrexprdata->ntermbvars[nlhdlrexprdata->nonoffterms] = nbvars;
   nlhdlrexprdata->nonoffterms++;

   return SCIP_OKAY;
}

/** adds an expression to the array of convex expressions */
static
SCIP_RETCODE addConvTerm(
   SCIP*                           scip,             /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLREXPRDATA*   nlhdlrexprdata,   /**< nonlinear handler expression data */
   SCIP_Real                       coef,             /**< coefficient of expr in the original expression */
   SCIP_CONSEXPR_EXPR*             expr              /**< expression to be added */
)
{
   int newsize;

   assert(scip != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(expr != NULL);

   if( nlhdlrexprdata->nconvterms + 1 > nlhdlrexprdata->convtermssize )
   {
      newsize = SCIPcalcMemGrowSize(scip, nlhdlrexprdata->nconvterms + 1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->convterms,  nlhdlrexprdata->convtermssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->convcoefs, nlhdlrexprdata->convtermssize, newsize) );
      nlhdlrexprdata->convtermssize = newsize;
   }
   assert(nlhdlrexprdata->nconvterms + 1 <= nlhdlrexprdata->convtermssize);

   nlhdlrexprdata->convcoefs[nlhdlrexprdata->nconvterms] = coef;
   nlhdlrexprdata->convterms[nlhdlrexprdata->nconvterms] = expr;
   nlhdlrexprdata->nconvterms++;

   return SCIP_OKAY;
}

/** constructs gradient linearization of a given expression and adds it to rowprep */
static
SCIP_RETCODE addGradientLinearisation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_ROWPREP*         rowprep,            /**< a rowprep where the linearization is stored */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression to be linearized */
   SCIP_Real             coef,               /**< coefficient of expr in the original expression */
   SCIP_SOL*             sol,                /**< solution to be separated */
   SCIP_Bool*            success             /**< indicates whether the linearization could be computed */
)
{
   SCIP_CONSEXPR_EXPR** varexprs;
   SCIP_Real constant;
   int i, v, nvars;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(rowprep != NULL);
   assert(expr != NULL);
   assert(success != NULL);

   /* compute gradient */
   SCIP_CALL( SCIPcomputeConsExprExprGradient(scip, conshdlr, expr, sol, 0) );

   /* gradient evaluation error -> skip */
   if( SCIPgetConsExprExprDerivative(expr) == SCIP_INVALID ) /*lint !e777*/
   {
      *success = FALSE;
      SCIPdebugMsg(scip, "gradient evaluation error for %p\n", (void*)expr);
      return SCIP_OKAY;
   }

   /* get g(x*) */
   constant = SCIPgetConsExprExprValue(expr);

   /* evaluation error or a too large constant -> skip */
   if( SCIPisInfinity(scip, REALABS(constant)) )
   {
      *success = FALSE;
      SCIPdebugMsg(scip, "evaluation error / too large value (%g) for %p\n", constant, (void*)expr);
      return SCIP_OKAY;
   }

   /* compute gradient cut */
   SCIP_CALL( SCIPallocBufferArray(scip, &varexprs, SCIPgetNTotalVars(scip)) );
   SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, expr, varexprs, &nvars) );
   for( i = 0; i < nvars; ++i )
   {
      SCIP_VAR* var;
      SCIP_Real derivative;
      SCIP_Real val;

      assert(varexprs[i] != NULL);
      assert(SCIPisConsExprExprVar(varexprs[i]));

      /* get the variable of the variable expression */
      var = SCIPgetConsExprExprVarVar(varexprs[i]);
      assert(var != NULL);

      /* get solution value */
      val = SCIPgetSolVal(scip, sol, var);

      /* avoid overhead of SCIPgetConsExprExprPartialDiff by accessing the derivative directly */
      derivative = SCIPgetConsExprExprDerivative(varexprs[i]);
      assert(SCIPgetConsExprExprPartialDiff(scip, conshdlr, expr, var) == derivative); /*lint !e777*/

      /* evaluation error or too large values -> skip */
      if( SCIPisInfinity(scip, REALABS(derivative * val)) )
      {
         *success = FALSE;
         SCIPdebugMsg(scip, "evaluation error / too large values (%g %g) for %s in %p\n", derivative, val,
                      SCIPvarGetName(var), (void*)expr);
         goto TERMINATE;
      }

      /* - grad(g(x*))_i x*_i */
      constant -= derivative * val;

      /* grad(g(x*))_i x_i */
      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, var, coef*derivative) );
   }

   /* add constant */
   SCIPaddRowprepConstant(rowprep, coef*constant);

 TERMINATE:
   for( v = 0; v < nvars; ++v )
   {
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &varexprs[v]) );
   }
   SCIPfreeBufferArray(scip, &varexprs);

   return SCIP_OKAY;
}

/** constructs perspective linearization of a given expression and adds it to rowprep */
static
SCIP_RETCODE addPerspectiveLinearisation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_HASHMAP*         scvars,             /**< hashmap linking semicontinuous vars to their data */
   SCIP_ROWPREP*         rowprep,            /**< a rowprep where the linearization is stored */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression to be linearized */
   SCIP_Real             coef,               /**< coefficient of expr */
   SCIP_VAR*             bvar,               /**< binary variable */
   SCIP_SOL*             sol,                /**< solution to be separated */
   SCIP_Bool*            success             /**< indicates whether the linearization could be computed */
   )
{
   SCIP_SOL* sol0;
   SCIP_Real* vals0;
   SCIP_CONSEXPR_EXPR** varexprs;
   SCIP_VAR** vars;
   SCIP_Real scalar_prod, fval, fval0;
   int nvars, v, pos;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(scvars != NULL);
   assert(rowprep != NULL);
   assert(expr != NULL);
   assert(bvar != NULL);
   assert(success != NULL);

   /* add the cut: auxvar >= (x - x0) \nabla f(sol) + (f(sol) - f(x0) - (sol - x0) \nabla f(sol)) z + f(x0),
    * where x is semicontinuous, z is binary and x0 is the value of x when z = 0 */

   SCIP_CALL( SCIPcreateSol(scip, &sol0, NULL) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varexprs, SCIPgetNTotalVars(scip)) );
   SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, expr, varexprs, &nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vals0, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "bvar = \n");
   SCIPprintVar(scip, bvar, NULL);
   SCIPdebugMsg(scip, NULL, "pexpr = \n");
   SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
#endif

   /* get x0 */
   for( v = 0; v < nvars; ++v )
   {
      SCIP_SCVARDATA* vardata;
      vars[v] = SCIPgetConsExprExprVarVar(varexprs[v]);
      vardata = (SCIP_SCVARDATA*)SCIPhashmapGetImage(scvars, (void*)vars[v]);

      /* find bvar in vardata->bvars */
      (void) SCIPsortedvecFindPtr((void**)vardata->bvars, SCIPvarComp, (void*)bvar, vardata->nbnds, &pos);
      assert(pos < vardata->nbnds);
      assert(vardata->bvars[pos] == bvar);

      vals0[v] = vardata->vals0[pos];
   }

   /* set x to x0 in sol0 */
   SCIP_CALL( SCIPsetSolVals(scip, sol0, nvars, vars, vals0) );

   /* get f(x0) */
   SCIP_CALL( SCIPevalConsExprExpr(scip, conshdlr, expr, sol0, 0) );
   fval0 = SCIPgetConsExprExprValue(expr);
   SCIP_CALL( SCIPfreeSol(scip, &sol0) );

   /* evaluation error or a too large constant -> skip */
   if( SCIPisInfinity(scip, REALABS(fval0)) )
   {
      *success = FALSE;
      SCIPdebugMsg(scip, "evaluation error / too large value (%g) for %p\n", fval0, (void*)expr);
      goto TERMINATE;
   }

   /* get f(sol) */
   SCIP_CALL( SCIPevalConsExprExpr(scip, conshdlr, expr, sol, 0) );
   fval = SCIPgetConsExprExprValue(expr);

   /* evaluation error or a too large constant -> skip */
   if( SCIPisInfinity(scip, REALABS(fval)) )
   {
      *success = FALSE;
      SCIPdebugMsg(scip, "evaluation error / too large value (%g) for %p\n", fval, (void*)expr);
      goto TERMINATE;
   }

   /* add (f(sol) - f(x0))z + f(x0) */
   SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, bvar, coef*(fval - fval0)) );
   SCIPaddRowprepConstant(rowprep, coef*fval0);

   /* compute gradient */
   SCIP_CALL( SCIPcomputeConsExprExprGradient(scip, conshdlr, expr, sol, 0) );

   /* gradient evaluation error -> skip */
   if( SCIPgetConsExprExprDerivative(expr) == SCIP_INVALID ) /*lint !e777*/
   {
      *success = FALSE;
      SCIPdebugMsg(scip, "gradient evaluation error for %p\n", (void*)expr);
      goto TERMINATE;
   }

   scalar_prod = 0.0;
   for( v = 0; v < nvars; ++v )
   {
      SCIP_VAR* var;
      var = SCIPgetConsExprExprVarVar(varexprs[v]);

      /* add xi f'xi(sol) */
      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, var, coef*SCIPgetConsExprExprPartialDiff(scip, conshdlr, expr, var)) );
      /* add -x0i f'xi(sol) */
      SCIPaddRowprepConstant(rowprep, -coef*vals0[v]*SCIPgetConsExprExprPartialDiff(scip, conshdlr, expr, var));

      /* compute -(soli - x0i) f'xi(sol) */
      scalar_prod -= (SCIPgetSolVal(scip, sol, var) - vals0[v])*SCIPgetConsExprExprPartialDiff(scip, conshdlr, expr, var);
   }

   /* add -(sol - x0) \nabla f(sol)) z */
   SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, bvar, coef*scalar_prod) );

 TERMINATE:
   SCIPfreeBufferArray(scip, &vars);
   SCIPfreeBufferArray(scip, &vals0);
   for( v = 0; v < nvars; ++v )
   {
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &varexprs[v]) );
   }
   SCIPfreeBufferArray(scip, &varexprs);

   return SCIP_OKAY;
}

/** adds an expression term either to convterms or to onoffterms */
static
SCIP_RETCODE addTerm(
   SCIP*                         scip,            /**< SCIP data structure */
   SCIP_CONSHDLR*                conshdlr,        /**< constraint handler */
   SCIP_CONSEXPR_NLHDLRDATA*     nlhdlrdata,      /**< nonlinear handler data */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata,  /**< nlhdlr expression data */
   SCIP_CONSEXPR_EXPR*           term,            /**< expression term to be added */
   SCIP_Real                     coef             /**< coefficient of the term in the parent expression */
   )
{
   SCIP_CONSEXPR_EXPR** varexprs;
   int nvars, v, nbvars, nbvars0;
   SCIP_Bool var_is_sc;
   SCIP_VAR* var;
   SCIP_SCVARDATA* scvdata;
   SCIP_VAR** expr_bvars;

   SCIP_CALL( SCIPallocBufferArray(scip, &varexprs, nlhdlrexprdata->nvarexprs) );
   SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, term, varexprs, &nvars) );

   /* a constant is not an on/off expression */
   if( nvars == 0 )
   {
      SCIP_CALL( addConvTerm(scip, nlhdlrexprdata, coef, term) );
      goto TERMINATE;
   }

   /* all variables of an on/off term should be semicontinuous */
   for( v = 0; v < nvars; ++v )
   {
      var = SCIPgetConsExprExprVarVar(varexprs[v]);
      SCIP_CALL( varIsSemicontinuous(scip, var, nlhdlrdata->scvars, &var_is_sc) );
      if( !var_is_sc )
      {
         SCIP_CALL( addConvTerm(scip, nlhdlrexprdata, coef, term) );
         goto TERMINATE;
      }
   }

   /* find common binary variables for all variables of children[c] */
   scvdata = (SCIP_SCVARDATA*)SCIPhashmapGetImage(nlhdlrdata->scvars, (void*)SCIPgetConsExprExprVarVar(varexprs[0]));
   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &expr_bvars, scvdata->bvars, scvdata->nbnds) );
   nbvars = scvdata->nbnds;
   nbvars0 = scvdata->nbnds;

   SCIPdebugMsg(scip, "\nArray intersection for vars %s", SCIPvarGetName(SCIPgetConsExprExprVarVar(varexprs[0])));
   for( v = 1; v < nvars; ++v )
   {
#ifdef SCIP_DEBUG
      SCIPinfoMessage(scip, NULL, "\n%s; ", SCIPvarGetName(SCIPgetConsExprExprVarVar(varexprs[v])));
#endif
      scvdata = (SCIP_SCVARDATA*)SCIPhashmapGetImage(nlhdlrdata->scvars, (void*)SCIPgetConsExprExprVarVar(varexprs[v]));
      SCIPcomputeArraysIntersectionPtr((void**)expr_bvars, nbvars, (void**)scvdata->bvars, scvdata->nbnds, SCIPvarComp, (void**)expr_bvars, &nbvars);

      /* if we have found out that the intersection is empty, term can be immediately added to convterms */
      if( nbvars == 0 )
      {
         SCIPfreeBlockMemoryArray(scip, &expr_bvars, nbvars0);
         SCIP_CALL( addConvTerm(scip, nlhdlrexprdata, coef, term) );
         goto TERMINATE;
      }
   }
   SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &expr_bvars, nbvars0, nbvars) );

#ifdef SCIP_DEBUG
   SCIPinfoMessage(scip, NULL, " is: ");
   for( v = 0; v < nbvars; ++v )
   {
      SCIPinfoMessage(scip, NULL, "%s; ", SCIPvarGetName(expr_bvars[v]));
   }
   SCIPdebugMsg(scip, "Adding on/off term: ");
   SCIPprintConsExprExpr(scip, conshdlr, term, NULL);
#endif

   /* if the term satisfies the requirements for g_i(x_i), add it to onoffterms */
   SCIP_CALL( addOnoffTerm(scip, conshdlr, nlhdlrexprdata, coef, term, expr_bvars, nbvars) );

 TERMINATE:
   for( v = 0; v < nvars; ++v )
   {
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &varexprs[v]) );
   }
   SCIPfreeBufferArray(scip, &varexprs);

   return SCIP_OKAY;
}

/** add the cut given by rowprep to sepastore */
static
SCIP_RETCODE addCut(
   SCIP*         scip,
   SCIP_CONS*    cons,
   SCIP_ROWPREP* rowprep,
   SCIP_SOL*     sol,
   double        mincutviolation,
   int*          ncuts,
   SCIP_RESULT*  result
   )
{
   SCIP_Bool success;
   SCIP_ROW* row;

   /* merge coefficients that belong to same variable */
   SCIPmergeRowprepTerms(scip, rowprep);

   SCIP_CALL( SCIPcleanupRowprep(scip, rowprep, sol, SCIP_CONSEXPR_CUTMAXRANGE, mincutviolation, NULL, &success) );

   /* if cut looks good (numerics ok and cutting off solution), then turn into row and add to sepastore */
   if( success )
   {
      SCIP_Bool infeasible;
      SCIP_CALL( SCIPgetRowprepRowCons(scip, &row, rowprep, cons) );
#ifdef SCIP_DEBUG
      SCIPdebugMsg(scip, "Separating sol point\n");
      for( int v = 0; v < nlhdlrexprdata->nvarexprs; ++v )
      {
         SCIP_VAR* var = SCIPgetConsExprExprVarVar(nlhdlrexprdata->varexprs[v]);
         SCIPwriteVarName(scip, NULL, var, TRUE);
         SCIPinfoMessage(scip, NULL, ": %f\n",  SCIPgetSolVal(scip, sol, var));
      }
      SCIPinfoMessage(scip, NULL, "by perspective cut ");
      SCIP_CALL( SCIPprintRow(scip, row, NULL) );
#endif

      SCIP_CALL( SCIPaddRow(scip, row, FALSE, &infeasible) );

      if( infeasible )
      {
         *result = SCIP_CUTOFF;
      }
      else
      {
         *result = SCIP_SEPARATED;
         ++*ncuts;
      }

      SCIP_CALL( SCIPreleaseRow(scip, &row) );
   }

   return SCIP_OKAY;
}

/** frees nlhdlrexprdata structure */
static
SCIP_RETCODE freeNlhdlrExprData(
   SCIP*                         scip,            /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata   /**< nlhdlr expression data */
   )
{
   int c;

   for( c = 0; c < nlhdlrexprdata->nonoffterms; ++c )
   {
      SCIPfreeBlockMemoryArray(scip, &(nlhdlrexprdata->termbvars[c]), nlhdlrexprdata->ntermbvars[c]);
   }
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->termbvars), nlhdlrexprdata->onofftermssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->ntermbvars), nlhdlrexprdata->onofftermssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->onoffcoefs), nlhdlrexprdata->onofftermssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->onoffterms), nlhdlrexprdata->onofftermssize);

   if( nlhdlrexprdata->varexprs != NULL )
   {
      for( c = 0; c < nlhdlrexprdata->nvarexprs; ++c )
      {
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &(nlhdlrexprdata->varexprs[c])) );
      }
      SCIPfreeBlockMemoryArray(scip, &nlhdlrexprdata->varexprs, nlhdlrexprdata->nvarexprs);
   }

   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->convcoefs), nlhdlrexprdata->convtermssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->convterms), nlhdlrexprdata->convtermssize);

   return SCIP_OKAY;
}

/*
 * Callback methods of nonlinear handler
 */

/** nonlinear handler copy callback */
static
SCIP_DECL_CONSEXPR_NLHDLRCOPYHDLR(nlhdlrCopyhdlrPerspective)
{ /*lint --e{715}*/
   assert(targetscip != NULL);
   assert(targetconsexprhdlr != NULL);
   assert(sourcenlhdlr != NULL);
   assert(strcmp(SCIPgetConsExprNlhdlrName(sourcenlhdlr), NLHDLR_NAME) == 0);

   SCIP_CALL( SCIPincludeConsExprNlhdlrPerspective(targetscip, targetconsexprhdlr) );

   return SCIP_OKAY;
}


/** callback to free data of handler */
static
SCIP_DECL_CONSEXPR_NLHDLRFREEHDLRDATA(nlhdlrFreehdlrdataPerspective)
{ /*lint --e{715}*/
   SCIP_HASHMAPENTRY* entry;
   SCIP_SCVARDATA* data;
   int c;

   if( (*nlhdlrdata)->scvars != NULL )
   {
      for( c = 0; c < SCIPhashmapGetNEntries((*nlhdlrdata)->scvars); ++c )
      {
         entry = SCIPhashmapGetEntry((*nlhdlrdata)->scvars, c);
         if( entry != NULL )
         {
            data = (SCIP_SCVARDATA*) SCIPhashmapEntryGetImage(entry);
            SCIPfreeBlockMemoryArray(scip, &data->vals0, data->bndssize);
            SCIPfreeBlockMemoryArray(scip, &data->bvars, data->bndssize);
            SCIPfreeBlockMemory(scip, &data);
         }
      }
      SCIPhashmapFree(&((*nlhdlrdata)->scvars));
   }

   SCIPfreeBlockMemory(scip, nlhdlrdata);

   return SCIP_OKAY;
}


/** callback to free expression specific data */
static
SCIP_DECL_CONSEXPR_NLHDLRFREEEXPRDATA(nlhdlrFreeExprDataPerspective)
{  /*lint --e{715}*/
   SCIP_CALL( freeNlhdlrExprData(scip, *nlhdlrexprdata) );
   SCIPfreeBlockMemory(scip, nlhdlrexprdata);

   return SCIP_OKAY;
}



/** callback to be called in initialization */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRINIT(nlhdlrInitPerspective)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}
#endif


/** callback to be called in deinitialization */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLREXIT(nlhdlrExitPerspective)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}
#endif


/** callback to detect structure in expression tree
 *
 * We are looking for expressions of the form: \sum\limits_{i=1}^p g_i(x_i) + g_0(x_0), where:
 *  each vector x_i has a single fixed value x^{off}_i when a binary var b_i is 0;
 *  g_i, i=1,..,p are nonlinear and either all convex or all concave;
 *  g_0 is either linear or has the same curvature as g_i, i=1,..,p;
 *  p != 0.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRDETECT(nlhdlrDetectPerspective)
{ /*lint --e{715}*/
   int c, nchildren;
   SCIP_CONSEXPR_EXPR** children;
   SCIP_Real* coefs;
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;

   nlhdlrdata = SCIPgetConsExprNlhdlrData(nlhdlr);

   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(expr != NULL);
   assert(enforcemethods != NULL);
   assert(enforcedbelow != NULL);
   assert(enforcedabove != NULL);
   assert(success != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(nlhdlrdata != NULL);

   *success = TRUE;

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "Called perspective detect, expr = %p: \n", expr);
   SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
#endif

   /* ignore sums */
   if( !DETECTSUM && SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrSum(conshdlr) ) /*lint !e506 !e774*/
   {
      *success = FALSE;
      return SCIP_OKAY;
   }

   if( SCIPgetConsExprExprCurvature(expr) == SCIP_EXPRCURV_UNKNOWN || SCIPgetConsExprExprCurvature(expr) == SCIP_EXPRCURV_LINEAR )
   {
      SCIPdebugMsg(scip, "curvature of expr %p is %s\n", expr, SCIPgetConsExprExprCurvature(expr) == SCIP_EXPRCURV_LINEAR ? "linear" : "unknown");
      *success = FALSE;
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPallocClearBlockMemory(scip, nlhdlrexprdata) );
   (*nlhdlrexprdata)->curvature = SCIPgetConsExprExprCurvature(expr);
   SCIPdebugMsg(scip, "expr %p is %s\n", expr, (*nlhdlrexprdata)->curvature == SCIP_EXPRCURV_CONVEX ? "convex" : "concave");

   SCIP_CALL( SCIPgetConsExprExprNVars(scip, conshdlr, expr, &(*nlhdlrexprdata)->nvarexprs) );
   if( nlhdlrdata->scvars == NULL )
   {
      SCIP_CALL( SCIPhashmapCreate(&(nlhdlrdata->scvars), SCIPblkmem(scip), (*nlhdlrexprdata)->nvarexprs) );
   }

   /* prepare the list of terms */
   if( SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrSum(conshdlr) )
   {
      children = SCIPgetConsExprExprChildren(expr);
      nchildren = SCIPgetConsExprExprNChildren(expr);
      coefs = SCIPgetConsExprExprSumCoefs(expr);
   }
   else
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &children, 1) );
      *children = expr;
      nchildren = 1;
      SCIP_CALL( SCIPallocBufferArray(scip, &coefs, 1) );
      *coefs = 1.0;
   }

   /* collect terms that satisfy the conditions for g_i(x_i) and the corresponding binary variables
    * all other terms are stored in convterms */
   for( c = 0; c < nchildren; ++c )
   {
      if( SCIPexprcurvMultiply(coefs[c], SCIPgetConsExprExprCurvature(children[c])) != (*nlhdlrexprdata)->curvature )
      {
         *success = FALSE;
         break;
      }
      SCIP_CALL( addTerm(scip, conshdlr, nlhdlrdata, *nlhdlrexprdata, children[c], coefs[c]) );
   }

   if( (*nlhdlrexprdata)->nonoffterms == 0 )
      *success = FALSE;

   if( SCIPgetConsExprExprHdlr(expr) != SCIPgetConsExprExprHdlrSum(conshdlr) )
   {
      SCIPfreeBufferArray(scip, &coefs);
      SCIPfreeBufferArray(scip, &children);
   }

   if( *success )
   {
      SCIPdebugMsg(scip, "\ndetected an on/off expr");

      /* depending on curvature, set enforcemethods */
      if( (*nlhdlrexprdata)->curvature == SCIP_EXPRCURV_CONVEX )
      {
         *enforcemethods |= SCIP_CONSEXPR_EXPRENFO_SEPABELOW;
         *enforcedbelow = TRUE;
         SCIPdebugMsg(scip, "detected expr to be convex -> can enforce expr <= auxvar\n");
      }
      else if( (*nlhdlrexprdata)->curvature == SCIP_EXPRCURV_CONCAVE )
      {
         *enforcemethods |= SCIP_CONSEXPR_EXPRENFO_SEPAABOVE;
         *enforcedabove = TRUE;
         SCIPdebugMsg(scip, "detected expr to be concave -> can enforce expr >= auxvar\n");
      }
      /* save varexprs to nlhdlrexprdata */
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*nlhdlrexprdata)->varexprs, (*nlhdlrexprdata)->nvarexprs) );
      SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, expr, (*nlhdlrexprdata)->varexprs, &(*nlhdlrexprdata)->nvarexprs) );
      assert(*nlhdlrexprdata != NULL);
   }
   else
   {
      SCIP_CALL( freeNlhdlrExprData(scip, *nlhdlrexprdata) );
      SCIPfreeBlockMemory(scip, nlhdlrexprdata);
      *nlhdlrexprdata = NULL;
   }

   return SCIP_OKAY;
}


/** auxiliary evaluation callback of nonlinear handler */
static
SCIP_DECL_CONSEXPR_NLHDLREVALAUX(nlhdlrEvalauxPerspective)
{ /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);
   assert(auxvalue != NULL);

   *auxvalue = SCIPgetConsExprExprValue(expr);

   return SCIP_OKAY;
}


/** callback to detect structure in expression tree */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRINITSEPA(nlhdlrInitSepaPerspective)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of perspective nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#endif


/** separation deinitialization method of a nonlinear handler (called during CONSEXITSOL) */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLREXITSEPA(nlhdlrExitSepaPerspective)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of perspective nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#endif


/** nonlinear handler separation callback
 *
 * Applies perspective linearization to on/off terms and gradient linearization to everything else.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRSEPA(nlhdlrSepaPerspective)
{ /*lint --e{715}*/
   SCIP_ROWPREP* rowprep;
   SCIP_VAR* auxvar;
   int i, j;
   SCIP_CONSEXPR_EXPR* pexpr;
   SCIP_Bool success;
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;
   SCIP_Real pcoef;
   SCIP_VAR** bvars;

   *result = SCIP_DIDNOTFIND;

   nlhdlrdata = SCIPgetConsExprNlhdlrData(nlhdlr);

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "sepa method of perspective nonlinear handler called for expr %p: ", expr);
   SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, expr, NULL) );
   SCIPinfoMessage(scip, NULL, "\n");
#endif

   assert(scip != NULL);
   assert(expr != NULL);
   assert(conshdlr != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(result != NULL);
   assert(ncuts != NULL);
   assert(nlhdlrdata != NULL);

   *ncuts = 0;

   /* if estimating on non-convex side, then do nothing */
   if( ( overestimate && nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONVEX) ||
       (!overestimate && nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONCAVE) )
   {
      SCIPdebugMsg(scip, "Estimating on non-convex side, do nothing\n");
      return SCIP_OKAY;
   }

   auxvar = SCIPgetConsExprExprAuxVar(expr);
   assert(auxvar != NULL);

   if( !MULTCUTS || SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrSum(conshdlr) ) /*lint !e506 !e774*/
   {
      SCIP_CALL( SCIPcreateRowprep(scip, &rowprep, overestimate ? SCIP_SIDETYPE_LEFT : SCIP_SIDETYPE_RIGHT, FALSE) );
      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, auxvar, -1.0) );

      if( SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrSum(conshdlr) )
         SCIPaddRowprepConstant(rowprep, SCIPgetConsExprExprSumConstant(expr));

      success = TRUE; /* think positive */

      /* handle convex terms */
      for( i = 0; i < nlhdlrexprdata->nconvterms && success; ++i )
      {
         SCIP_CALL( addGradientLinearisation(scip, conshdlr, rowprep, nlhdlrexprdata->convterms[i], nlhdlrexprdata->convcoefs[i], sol, &success) );
      }

      /* handle on/off terms */
      for( i = 0; i < nlhdlrexprdata->nonoffterms && success; ++i )
      {
         SCIP_VAR* bvar;
         SCIP_Real minbval = 1, bval;

         /* heuristically choose the most promising binary variable (one closest to 0) */
         pexpr = nlhdlrexprdata->onoffterms[i];
         pcoef = nlhdlrexprdata->onoffcoefs[i];
         bvars = nlhdlrexprdata->termbvars[i];
         bvar = bvars[0];
         for( j = 1; j < nlhdlrexprdata->ntermbvars[i]; ++j)
         {
            bval = SCIPgetSolVal(scip, sol, bvars[j]);
            if( bvars[j] != NULL && bval < minbval )
            {
               minbval = bval;
               bvar = bvars[j];
            }
         }
         SCIP_CALL( addPerspectiveLinearisation(scip, conshdlr, nlhdlrdata->scvars, rowprep, pexpr, pcoef, bvar, sol, &success) );
      }

      if( success )
      {
         SCIP_CALL( addCut(scip, cons, rowprep, sol, mincutviolation, ncuts, result) );
      }

      SCIPfreeRowprep(scip, &rowprep);
   }
   else /* cuts for every suitable binary variable have been requested and expr is not a sum */
   {
      assert(nlhdlrexprdata->nonoffterms == 1);

      /* generate one cut for each binary variable */
      for( i = 0; i < nlhdlrexprdata->ntermbvars[0]; ++i )
      {
         SCIP_CALL( SCIPcreateRowprep(scip, &rowprep, overestimate ? SCIP_SIDETYPE_LEFT : SCIP_SIDETYPE_RIGHT, FALSE) );
         SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, auxvar, -1.0) );
         success = TRUE; /* think positive */

         SCIP_CALL( addPerspectiveLinearisation(scip, conshdlr, nlhdlrdata->scvars, rowprep, expr, 1.0, nlhdlrexprdata->termbvars[0][i], sol, &success) );

         if( success )
         {
            SCIP_CALL( addCut(scip, cons, rowprep, sol, mincutviolation, ncuts, result) );
         }

         SCIPfreeRowprep(scip, &rowprep);
         if( *result == SCIP_CUTOFF )
            break;
      }
   }

   return SCIP_OKAY;
}


/** nonlinear handler under/overestimation callback */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRESTIMATE(nlhdlrEstimatePerspective)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of perspective nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#endif


/** nonlinear handler interval evaluation callback */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRINTEVAL(nlhdlrIntevalPerspective)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of perspective nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#endif


/** nonlinear handler callback for reverse propagation */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP(nlhdlrReversepropPerspective)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of perspective nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#endif


/** nonlinear handler callback for branching scores */
static
SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE(nlhdlrBranchscorePerspective)
{ /*lint --e{715}*/
   SCIP_Real violation;
   int i;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprCurvature(expr) == SCIP_EXPRCURV_CONVEX || SCIPgetConsExprExprCurvature(expr) == SCIP_EXPRCURV_CONCAVE);
   assert(SCIPgetConsExprExprAuxVar(expr) != NULL);
   assert(auxvalue == SCIPgetConsExprExprValue(expr)); /* given auxvalue should have been computed by nlhdlrEvalAuxConvex */  /*lint !e777*/
   assert(nlhdlrexprdata != NULL);
   assert(nlhdlrexprdata->varexprs != NULL);
   assert(nlhdlrexprdata->nvarexprs > 0);
   assert(success != NULL);

   *success = FALSE;

   /* we separate only convex functions here, so there should be little use for branching
    * if violations are small or there are numerical issues, then we will not have generated a cut, though
    * in that case, we will still branch, that is, register branchscores for all depending var exprs
    */

   /* compute violation */
   if( auxvalue == SCIP_INVALID ) /*lint !e777*/
      violation = SCIPinfinity(scip); /* evaluation error -> we should branch */
   else if( SCIPgetConsExprExprCurvature(expr) == SCIP_EXPRCURV_CONVEX  )
      violation = auxvalue - SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(expr));
   else
      violation = SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(expr)) - auxvalue;

   /* if violation is not on the side that we need to enforce, then no need for branching */
   if( violation <= 0.0 )
      return SCIP_OKAY;

   /* TODO try to figure out which variables appear linear and skip them here */
   for( i = 0; i < nlhdlrexprdata->nvarexprs; ++i )
   {
      assert(nlhdlrexprdata->varexprs[i] != NULL);
      assert(SCIPisConsExprExprVar(nlhdlrexprdata->varexprs[i]));

      SCIPaddConsExprExprBranchScore(scip, nlhdlrexprdata->varexprs[i], brscoretag, violation);
   }

   *success = TRUE;

   return SCIP_OKAY;
}

/** nonlinear handler callback for reformulation */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRREFORMULATE(nlhdlrReformulatePerspective)
{ /*lint --e{715}*/

   /* TODO detect structure */

   /* TODO create expression and store it in refexpr */

   /* set refexpr to expr and capture it if no reformulation is possible */
   *refexpr = expr;
   SCIPcaptureConsExprExpr(*refexpr);

   return SCIP_OKAY;
}
#endif

/*
 * nonlinear handler specific interface methods
 */

/** includes Perspective nonlinear handler to consexpr */
SCIP_RETCODE SCIPincludeConsExprNlhdlrPerspective(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;
   SCIP_CONSEXPR_NLHDLR* nlhdlr;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);

   /* create nonlinear handler data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &nlhdlrdata) );
   BMSclearMemory(nlhdlrdata);


   SCIP_CALL( SCIPincludeConsExprNlhdlrBasic(scip, consexprhdlr, &nlhdlr, NLHDLR_NAME, NLHDLR_DESC, NLHDLR_PRIORITY, nlhdlrDetectPerspective, nlhdlrEvalauxPerspective, nlhdlrdata) );
   assert(nlhdlr != NULL);

   SCIPsetConsExprNlhdlrCopyHdlr(scip, nlhdlr, nlhdlrCopyhdlrPerspective);
   SCIPsetConsExprNlhdlrFreeHdlrData(scip, nlhdlr, nlhdlrFreehdlrdataPerspective);
   SCIPsetConsExprNlhdlrFreeExprData(scip, nlhdlr, nlhdlrFreeExprDataPerspective);
   SCIPsetConsExprNlhdlrSepa(scip, nlhdlr, NULL, nlhdlrSepaPerspective, NULL, NULL);
   SCIPsetConsExprNlhdlrBranchscore(scip, nlhdlr, nlhdlrBranchscorePerspective);

   return SCIP_OKAY;
}
