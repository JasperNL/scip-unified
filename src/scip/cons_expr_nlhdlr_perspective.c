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
#include "scip/scip_sol.h"
#include "scip/cons_expr_iterator.h"
#include "struct_cons_expr.h"
#include "scip/cons_expr_sum.h"

/* fundamental nonlinear handler properties */
#define NLHDLR_NAME               "perspective"
#define NLHDLR_DESC               "perspective handler for expressions"
#define NLHDLR_DETECTPRIORITY     -20 /**< detect last so that to make use of what other handlers detected */
#define NLHDLR_ENFOPRIORITY       125 /**< enforce first because perspective cuts are always stronger */

#define DEFAULT_DETECTSUM         FALSE /**< TODO this is not used currently */
#define DEFAULT_MULTCUTS          TRUE  /**< TODO this is not used currently */
#define DEFAULT_MAXPROPROUNDS     1     /**< maximal number of propagation rounds in probing */

/*
 * Data structures
 */

/** data structure to store information of a semicontinuous variable */
struct SCVarData
{
   SCIP_Real*            vals0;              /**< values of the variable when the corresponding bvars[i] = 0 */
   SCIP_Real*            lbs1;               /**< global lower bounds of the variable when the corresponding bvars[i] = 1 */
   SCIP_Real*            ubs1;               /**< global upper bounds of the variable when the corresponding bvars[i] = 1 */
   SCIP_VAR**            bvars;              /**< the binary variables on which the variable domain depends */
   int                   nbnds;              /**< number of suitable on/off bounds the var has */
   int                   bndssize;           /**< size of the arrays */
};
typedef struct SCVarData SCVARDATA;

/** nonlinear handler expression data */
struct SCIP_ConsExpr_NlhdlrExprData
{
#ifdef SCIP_DISABLED_CODE
   SCIP_EXPRCURV         curvature;          /**< curvature of the expression */

   SCIP_CONSEXPR_EXPR**  onoffterms;         /**< on/off terms for which we apply perspective cuts */
   SCIP_Real*            onoffcoefs;         /**< coefficients of onoffterms */
   int                   nonoffterms;        /**< number of on/off expressions */
   int                   onofftermssize;     /**< size of arrays describing on/off terms */
   SCIP_VAR***           termbvars;          /**< binary vars associated with onoffterms */
   int*                  ntermbvars;         /**< number of binary variables for each term */

   SCIP_CONSEXPR_EXPR**  convterms;          /**< convex terms for which we apply gradient cuts */
   SCIP_Real*            convcoefs;          /**< coefficients of convterms */
   int                   nconvterms;         /**< number of convterms */
   int                   convtermssize;      /**< size of the convterms array */
#endif

   SCIP_Real*            exprvals0;          /**< 'off' values of the expression for each indicator variable */
   SCIP_CONSEXPR_EXPR**  varexprs;           /**< variable expressions */
   int                   nvarexprs;          /**< total number of variable expressions */
   SCIP_VAR**            indicators;         /**< all indicator variables for the expression */
   int                   nindicators;        /**< number of indicator variables */
};

/** nonlinear handler data */
struct SCIP_ConsExpr_NlhdlrData
{
   SCIP_HASHMAP*         scvars;             /**< maps semicontinuous variables to their on/off bounds */

   /* parameters */
   SCIP_Bool             detectsum;          /**< whether to run detection when the root of an expression is a sum */
   SCIP_Bool             multcuts;           /**< whether to add cuts for all suitable indicator variables */
   int                   maxproprounds;      /**< maximal number of propagation rounds in probing */
};

/*
 * Local methods
 */

/** adds an indicator to the data of a semicontinuous variable */
static
SCIP_RETCODE addSCVarIndicator(
   SCIP*                 scip,               /**< SCIP data structure */
   SCVARDATA*            scvdata,            /**< semicontinuous variable data */
   SCIP_VAR*             indicator,          /**< indicator to be added */
   SCIP_Real             val0,               /**< value of the variable when indicator = 0 */
   SCIP_Real             lb1,                /**< lower bound of the variable when indicator = 1 */
   SCIP_Real             ub1                 /**< upper bound of the variable when indicator = 1 */
   )
{
   int newsize;
   int i;
   SCIP_Bool found;
   int pos;

   assert(scvdata != NULL);
   assert(indicator != NULL);

   /* ensure sizes */
   if( scvdata->nbnds + 1 > scvdata->bndssize )
   {
      newsize = SCIPcalcMemGrowSize(scip, scvdata->nbnds + 1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->bvars, scvdata->bndssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->vals0, scvdata->bndssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->lbs1, scvdata->bndssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &scvdata->ubs1, scvdata->bndssize, newsize) );
      scvdata->bndssize = newsize;
   }
   assert(scvdata->nbnds + 1 <= scvdata->bndssize);

   /* find the position where to insert */
   found = SCIPsortedvecFindPtr((void**)scvdata->bvars, SCIPvarComp, (void*)indicator, scvdata->nbnds, &pos);

   if( found )
      return SCIP_OKAY;

   /* move entries if needed */
   for( i = scvdata->nbnds; i > pos; --i )
   {
      scvdata->bvars[i] = scvdata->bvars[i-1];
      scvdata->vals0[i] = scvdata->vals0[i-1];
      scvdata->lbs1[i] = scvdata->lbs1[i-1];
      scvdata->ubs1[i] = scvdata->ubs1[i-1];
   }

   scvdata->bvars[pos] = indicator;
   scvdata->vals0[pos] = val0;
   scvdata->lbs1[pos] = lb1;
   scvdata->ubs1[pos] = ub1;
   ++scvdata->nbnds;

   return SCIP_OKAY;
}

/** find scvardata of var and position of indicator in it
 *
 *  If indicator is not there, returns NULL.
 */
static
SCVARDATA* getSCVarDataInd(
   SCIP_HASHMAP*         scvars,             /**< hashmap linking variables to scvardata */
   SCIP_VAR*             var,                /**< variable */
   SCIP_VAR*             indicator,          /**< indicator variable */
   int*                  pos                 /**< pointer to store the position of indicator */
   )
{
   SCIP_Bool exists;
   SCVARDATA* scvdata;

   assert(var != NULL);
   assert(scvars != NULL);
   assert(indicator != NULL);

   scvdata = (SCVARDATA*) SCIPhashmapGetImage(scvars, (void*)var);
   if( scvdata != NULL )
   {
      /* look for the indicator variable */
      exists = SCIPsortedvecFindPtr((void**)scvdata->bvars, SCIPvarComp, (void*)indicator, scvdata->nbnds, pos);
      if( !exists )
         return NULL;

      return scvdata;
   }

   return NULL;
}

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
   SCIP_VAR*             indicator,          /**< indicator variable which var should depend on (NULL if doesn't matter) */
   SCIP_Real*            val0,               /**< buffer to store value of var when indicator == 0 (NULL if not interested) */
   SCIP_Bool*            result              /**< buffer to store whether var is semicontinuous */
   )
{
   SCIP_Real lb0;
   SCIP_Real ub0;
   SCIP_Real lb1;
   SCIP_Real ub1;
   SCIP_Real glb;
   SCIP_Real gub;
   SCIP_Bool exists;
   int c;
   int pos;
   SCIP_VAR** vlbvars;
   SCIP_VAR** vubvars;
   SCIP_Real* vlbcoefs;
   SCIP_Real* vubcoefs;
   SCIP_Real* vlbconstants;
   SCIP_Real* vubconstants;
   int nvlbs;
   int nvubs;
   SCVARDATA* scvdata;
   SCIP_VAR* bvar;

   assert(scip != NULL);
   assert(var != NULL);
   assert(scvars != NULL);
   assert(result != NULL);

   scvdata = (SCVARDATA*) SCIPhashmapGetImage(scvars, (void*)var);
   if( scvdata != NULL )
   {
      *result = TRUE;

      if( indicator != NULL )
      { /* if the indicator variable matters, look for it */
         exists = SCIPsortedvecFindPtr((void**)scvdata->bvars, SCIPvarComp, (void*)indicator, scvdata->nbnds, &pos);
         if( !exists )
            *result = FALSE;
         else if( val0 != NULL )
            *val0 = scvdata->vals0[pos];
      }

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

   *result = FALSE;

   /* Scan through lower bounds; for each binary vlbvar save the corresponding lb0 and lb1.
    * Then check if there is an upper bound with this vlbvar and save ub0 and ub1.
    * If the found bounds imply that the var value is fixed to some val0 when vlbvar = 0,
    * save vlbvar and val0 to scvdata.
    */
   for( c = 0; c < nvlbs; ++c )
   {
      if( SCIPvarGetType(vlbvars[c]) != SCIP_VARTYPE_BINARY )
         continue;

      SCIPdebugMsg(scip, "var <%s>[%f, %f] lower bound: %f <%s> %+f", SCIPvarGetName(var), glb, gub, vlbcoefs[c], SCIPvarGetName(vlbvars[c]), vlbconstants[c]);

      bvar = vlbvars[c];

      lb0 = MAX(vlbconstants[c], glb);
      lb1 = MAX(vlbconstants[c] + vlbcoefs[c], glb);

      /* look for bvar in vubvars */
      if( vubvars != NULL )
         exists = SCIPsortedvecFindPtr((void**)vubvars, SCIPvarComp, bvar, nvubs, &pos);
      else
         exists = FALSE;
      if( exists )
      { /*lint --e{644}*/
         SCIPdebugMsgPrint(scip, ", upper bound: %f <%s> %+f", vubcoefs[pos], SCIPvarGetName(vubvars[pos]), vubconstants[pos]); /*lint !e613*/

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
      SCIPdebugMsgPrint(scip, " -> <%s> in [%f, %f] (off), [%f, %f] (on)\n", SCIPvarGetName(var), lb0, ub0, lb1, ub1);
      if( lb0 == ub0 && (lb0 != lb1 || ub0 != ub1) ) /*lint !e777*/
      {
         if( scvdata == NULL )
         {
            SCIP_CALL( SCIPallocClearBlockMemory(scip, &scvdata) );
         }

         addSCVarIndicator(scip, scvdata, bvar, lb0, lb1, ub1);
      }
   }

   /* look for vubvars that have not been processed yet */
   assert(vubvars != NULL || nvubs == 0);
   for( c = 0; c < nvubs; ++c )
   {
      if( SCIPvarGetType(vubvars[c]) != SCIP_VARTYPE_BINARY)  /*lint !e613*/
         continue;

      bvar = vubvars[c];  /*lint !e613*/

      /* skip vars that are in vlbvars */
      if( vlbvars != NULL && SCIPsortedvecFindPtr((void**)vlbvars, SCIPvarComp, bvar, nvlbs, &pos) )
         continue;

      SCIPdebugMsg(scip, "var <%s>[%f, %f] upper bound: %f <%s> %+f",
         SCIPvarGetName(var), glb, gub, vubcoefs[c], SCIPvarGetName(vubvars[c]), vubconstants[c]);  /*lint !e613*/

      lb0 = glb;
      lb1 = glb;
      ub0 = MIN(vubconstants[c], gub);
      ub1 = MIN(vubconstants[c] + vubcoefs[c], gub);

      /* the 'off' domain of a semicontinuous var should reduce to a single point and be different from the 'on' domain */
      SCIPdebugMsgPrint(scip, " -> <%s> in [%f, %f] (off), [%f, %f] (on)\n", SCIPvarGetName(var), lb0, ub0, lb1, ub1);
      if( lb0 == ub0 && (lb0 != lb1 || ub0 != ub1) ) /*lint !e777*/
      {
         if( scvdata == NULL )
         {
            SCIP_CALL( SCIPallocClearBlockMemory(scip, &scvdata) );
         }

         addSCVarIndicator(scip, scvdata, bvar, lb0, lb1, ub1);
      }
   }

   if( scvdata != NULL )
   {
#ifdef SCIP_DEBUG
      SCIPdebugMsg(scip, "var <%s> has global bounds [%f, %f] and the following on/off bounds:\n", SCIPvarGetName(var), glb, gub);
      for( c = 0; c < scvdata->nbnds; ++c )
      {
         SCIPdebugMsg(scip, " c = %d, bvar <%s>: val0 = %f\n", c, SCIPvarGetName(scvdata->bvars[c]), scvdata->vals0[c]);
      }
#endif
      SCIP_CALL( SCIPhashmapInsert(scvars, var, scvdata) );
      *result = TRUE;
   }

   return SCIP_OKAY;
}

/* TODO keeping the unused functions for now, will decide if they are needed later */
#ifdef SCIP_DISABLED_CODE
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
      SCVARDATA* vardata;
      vars[v] = SCIPgetConsExprExprVarVar(varexprs[v]);
      vardata = (SCVARDATA*)SCIPhashmapGetImage(scvars, (void*)vars[v]);

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

   /* TODO it should not be necessary to reevaluate in sol, cons_expr should have done that already */
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
#endif

/* checks if an expression is semicontinuous
 *
 * An expression is semicontinuous if all of its variables are semicontinuous
 * and share at least one common indicator variable
 */
static
SCIP_RETCODE exprIsSemicontinuous(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata,     /**< nonlinear handler data */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata, /**< nlhdlr expression data */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_Bool*            res                 /**< buffer to store whether the expression is semicontinuous */
   )
{
   int v;
   int nvarexprs;
   SCIP_CONSEXPR_EXPR** varexprs;
   SCIP_Bool var_is_sc;
   SCVARDATA* scvdata;
   SCIP_VAR* var;
   int nindicators;
   int nbnds0;
   int c;
   SCIP_VAR** indicators;

   *res = FALSE;

   nvarexprs = nlhdlrexprdata->nvarexprs;
   varexprs = nlhdlrexprdata->varexprs;

   /* constant expression is not semicontinuous */
   if( nvarexprs == 0 )
   {
      return SCIP_OKAY;
   }

   if( SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrSum(conshdlr) )
   {
      for( c = 0; c < SCIPgetConsExprExprNChildren(expr); ++c )
      {
         SCIP_CONSEXPR_EXPR* child = SCIPgetConsExprExprChildren(expr)[c];
         SCIP_CONSEXPR_EXPR** childvarexprs;
         int nchildvarexprs;
         SCIP_Bool issc;

         SCIP_CALL( SCIPallocBufferArray(scip, &childvarexprs, nvarexprs) );
         SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, child, childvarexprs, &nchildvarexprs) );

         issc = TRUE;

         /* all nonlinear variables of a sum on/off term should be semicontinuous */
         for( v = 0; v < nchildvarexprs; ++v )
         {
            SCIP_Bool childacceptable;

            var = SCIPgetConsExprExprVarVar(childvarexprs[v]);
            SCIP_CALL( varIsSemicontinuous(scip, var, nlhdlrdata->scvars, NULL, NULL, &var_is_sc) );

            childacceptable = var_is_sc || SCIPgetConsExprExprAuxVar(child) != NULL;

            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &childvarexprs[v]) );

            if( !childacceptable )
            {
               issc = FALSE;
               break;
            }
         }

         SCIPfreeBufferArray(scip, &childvarexprs);
         if( !issc )
            return SCIP_OKAY;
      }
   }
   else
   {
      /* all variables of a non-sum on/off term should be semicontinuous */
      for( v = 0; v < nvarexprs; ++v )
      {
         var = SCIPgetConsExprExprVarVar(varexprs[v]);
         SCIP_CALL( varIsSemicontinuous(scip, var, nlhdlrdata->scvars, NULL, NULL, &var_is_sc) );
         if( !var_is_sc )
            return SCIP_OKAY;
      }
   }

   /* look for common binary variables for all variables of the expression */

   indicators = NULL;
   nindicators = 0;

   SCIPdebugMsg(scip, "Array intersection for vars %s, *nbvars = %d\n", SCIPvarGetName(SCIPgetConsExprExprVarVar(varexprs[0])), nindicators);
   for( v = 0; v < nvarexprs; ++v )
   {
#ifdef SCIP_DEBUG
      SCIPinfoMessage(scip, NULL, "\n%s; ", SCIPvarGetName(SCIPgetConsExprExprVarVar(varexprs[v])));
#endif
      scvdata = (SCVARDATA*)SCIPhashmapGetImage(nlhdlrdata->scvars, (void*)SCIPgetConsExprExprVarVar(varexprs[v]));

      if( scvdata == NULL )
         continue;

      if( indicators == NULL )
      {
         SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &indicators, scvdata->bvars, scvdata->nbnds) );
         nbnds0 = scvdata->nbnds;
         nindicators = nbnds0;
      }
      else
      {
         SCIPcomputeArraysIntersectionPtr((void**)indicators, nindicators, (void**)scvdata->bvars, scvdata->nbnds,
               SCIPvarComp, (void**)indicators, &nindicators);
      }

      /* if we have found out that the intersection is empty, expr is not semicontinuous */
      if( indicators != NULL && nindicators == 0 )
      {
         SCIPfreeBlockMemoryArray(scip, &indicators, nbnds0);
         return SCIP_OKAY;
      }
   }

   assert(indicators != NULL);
   assert(nindicators > 0 && nindicators <= nbnds0);

   if( nindicators < nbnds0 )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &indicators, nbnds0, nindicators) );
   }

   nlhdlrexprdata->indicators = indicators;
   nlhdlrexprdata->nindicators = nindicators;
   *res = TRUE;

   return SCIP_OKAY;
}

/** add the cut given by rowprep to sepastore */
static
SCIP_RETCODE addCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< expression constraint */
   SCIP_ROWPREP*         rowprep,            /**< cut to be added */
   SCIP_SOL*             sol,                /**< solution to be separated */
   SCIP_RESULT*          result              /**< pointer to store result */
   )
{
   SCIP_Bool success;
   SCIP_ROW* row;

   /* merge coefficients that belong to same variable */
   SCIPmergeRowprepTerms(scip, rowprep);

   SCIP_CALL( SCIPcleanupRowprep(scip, rowprep, sol, SCIP_CONSEXPR_CUTMAXRANGE, SCIPgetLPFeastol(scip), NULL, &success) );

   /* if cut looks good (numerics ok and cutting off solution), then turn into row and add to sepastore */
   if( success )
   {
      SCIP_Bool infeasible;

      SCIP_CALL( SCIPgetRowprepRowCons(scip, &row, rowprep, cons) );

      SCIP_CALL( SCIPaddRow(scip, row, FALSE, &infeasible) );

      if( infeasible )
      {
         *result = SCIP_CUTOFF;
      }
      else
      {
         *result = SCIP_SEPARATED;
      }

      SCIP_CALL( SCIPreleaseRow(scip, &row) );
   }

   return SCIP_OKAY;
}

/** frees nlhdlrexprdata structure */
static
SCIP_RETCODE freeNlhdlrExprData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata /**< nlhdlr expression data */
   )
{
   int c;

   if( nlhdlrexprdata->nindicators != 0 )
   {
      assert(nlhdlrexprdata->indicators != NULL);
      SCIPfreeBlockMemoryArray(scip, &(nlhdlrexprdata->indicators), nlhdlrexprdata->nindicators);
      SCIPfreeBlockMemoryArray(scip, &(nlhdlrexprdata->exprvals0), nlhdlrexprdata->nindicators);
   }

   if( nlhdlrexprdata->varexprs != NULL )
   {
      for( c = 0; c < nlhdlrexprdata->nvarexprs; ++c )
      {
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &(nlhdlrexprdata->varexprs[c])) );
      }
      SCIPfreeBlockMemoryArray(scip, &nlhdlrexprdata->varexprs, nlhdlrexprdata->nvarexprs);
   }

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
   SCVARDATA* data;
   int c;

   if( (*nlhdlrdata)->scvars != NULL )
   {
      for( c = 0; c < SCIPhashmapGetNEntries((*nlhdlrdata)->scvars); ++c )
      {
         entry = SCIPhashmapGetEntry((*nlhdlrdata)->scvars, c);
         if( entry != NULL )
         {
            data = (SCVARDATA*) SCIPhashmapEntryGetImage(entry);
            SCIPfreeBlockMemoryArray(scip, &data->ubs1, data->bndssize);
            SCIPfreeBlockMemoryArray(scip, &data->lbs1, data->bndssize);
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

/* remove an indicator from nonlinear expression data */
static
void removeExprIndicator(
  SCIP_CONSEXPR_NLHDLREXPRDATA* nlexprdata,  /**< nonlinear expression data */
  int                    pos                 /**< position of the indicator */
  )
{
   int i;

   assert(pos >= 0 && pos < nlexprdata->nindicators);

   for( i = pos; i < nlexprdata->nindicators - 1; ++i )
   {
      nlexprdata->indicators[i] = nlexprdata->indicators[i+1];
   }

   --nlexprdata->nindicators;
}

/** computes the 'off' value of the expression and the 'off' values of
  * semicontinuous auxiliary variables for each indicator variable
  */
static
SCIP_RETCODE computeOffValues(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_CONSEXPR_NLHDLRDATA* hdlrdata,       /**< nonlinear handler data */
   SCIP_CONSEXPR_NLHDLREXPRDATA* exprdata,   /**< nonlinear expression data */
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_SOL* sol;
   int i;
   int v;
   SCIP_Real* vals0;
   SCIP_VAR** vars;
   SCIP_Bool var_is_sc;
   SCVARDATA* scvdata;
   SCIP_VAR* auxvar;
   SCIP_CONSEXPR_EXPR* curexpr;

   assert(expr != NULL);

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(exprdata->exprvals0), exprdata->nindicators) );

   SCIP_CALL( SCIPcreateSol(scip, &sol, NULL) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, exprdata->nvarexprs) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vals0, exprdata->nvarexprs) );
   SCIP_CALL( SCIPexpriteratorCreate(&it, conshdlr, SCIPblkmem(scip)) );

   for( v = 0; v < exprdata->nvarexprs; ++v )
   {
      vars[v] = SCIPgetConsExprExprVarVar(exprdata->varexprs[v]);
   }

   for( i = 0; i < exprdata->nindicators; ++i )
   {
      /* set sol to the off value of all expr vars for this indicator */
      for( v = 0; v < exprdata->nvarexprs; ++v )
      {
         SCIP_CALL( varIsSemicontinuous(scip, vars[v], hdlrdata->scvars, exprdata->indicators[i], &vals0[v], &var_is_sc) );

         /* set vals0[v] = 0 if var is non-sc - then it will not contribute to exprvals0[i] */
         if( !var_is_sc )
            vals0[v] = 0.0;
      }
      SCIPsetSolVals(scip, sol, exprdata->nvarexprs, vars, vals0);
      SCIP_CALL( SCIPevalConsExprExpr(scip, conshdlr, expr, sol, 0) );

      if( SCIPgetConsExprExprValue(expr) == SCIP_INVALID )
      {
         SCIPdebugMsg(scip, "expression evaluation failed for %p, removing the indicator\n", (void*)expr);
         removeExprIndicator(exprdata, i);
         continue;
      }

      exprdata->exprvals0[i] = SCIPgetConsExprExprValue(expr);

      /* iterate through the expression and create scvdata for all aux vars */
      SCIP_CALL( SCIPexpriteratorInit(it, expr, SCIP_CONSEXPRITERATOR_DFS, FALSE) );
      curexpr = SCIPexpriteratorGetCurrent(it);

      while( !SCIPexpriteratorIsEnd(it) )
      {
         if( curexpr->auxvar != NULL )
         {
            /* we know that all vars are sc with respect to exprdata->indicators; it remains to:
             * - get or create the scvdata structure
             * - add it to scvars hashmap
             * - find the expr's off value
             * - add the indicator and off value to scvdata
             */
            auxvar = SCIPgetConsExprExprAuxVar(curexpr);

            scvdata = (SCVARDATA*) SCIPhashmapGetImage(hdlrdata->scvars, (void*)auxvar);
            if( scvdata == NULL )
            {
               SCIP_CALL( SCIPallocClearBlockMemory(scip, &scvdata) );
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &scvdata->bvars,  exprdata->nindicators) );
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &scvdata->vals0, exprdata->nindicators) );
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &scvdata->lbs1, exprdata->nindicators) );
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &scvdata->ubs1, exprdata->nindicators) );
               scvdata->bndssize = exprdata->nindicators;
               SCIP_CALL( SCIPhashmapInsert(hdlrdata->scvars, auxvar, scvdata) );
            }

            SCIP_CALL( addSCVarIndicator(scip, scvdata, exprdata->indicators[i], SCIPgetConsExprExprValue(curexpr),
                     SCIPvarGetLbGlobal(auxvar), SCIPvarGetUbGlobal(auxvar)) );
         }

         curexpr = SCIPexpriteratorGetNext(it);
      }
   }

   SCIPexpriteratorFree(&it);
   SCIPfreeBufferArray(scip, &vals0);
   SCIPfreeBufferArray(scip, &vars);
   SCIPfreeSol(scip, &sol);

   return SCIP_OKAY;
}

/** callback to detect structure in expression tree
 *
 *  We are looking for expressions g(x), where x is a vector of semicontinuous variables that all share at least one
 *  indicator variable.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRDETECT(nlhdlrDetectPerspective)
{ /*lint --e{715}*/
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

   *success = FALSE;

   /* do not run in presolve, as we only do separation */
   if( SCIPgetStage(scip) <= SCIP_STAGE_INITSOLVE )
   {
      return SCIP_OKAY;
   }

   if( SCIPgetNBinVars(scip) == 0 )
   {
      SCIPdebugMsg(scip, "problem has no binary variables, not running perspective detection\n");
      return SCIP_OKAY;
   }

   /* some other nonlinear handler should be able to separate */
   if( !(*enforcemethods & SCIP_CONSEXPR_EXPRENFO_SEPABELOW) && !(*enforcemethods & SCIP_CONSEXPR_EXPRENFO_SEPAABOVE) )
   {
      SCIPdebugMsg(scip, "\nno enforcement method, exiting detect");
      return SCIP_OKAY;
   }

#ifdef SCIP_DEBUG
      SCIPdebugMsg(scip, "Called perspective detect, expr = %p: ", expr);
   SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
   SCIPdebugMsgPrint(scip, "\n");
#endif

   /* allocate memory */
   SCIP_CALL( SCIPallocClearBlockMemory(scip, nlhdlrexprdata) );
   if( nlhdlrdata->scvars == NULL )
   {
      SCIP_CALL( SCIPhashmapCreate(&(nlhdlrdata->scvars), SCIPblkmem(scip), SCIPgetNVars(scip)) );
   }

   /* save varexprs to nlhdlrexprdata */
   SCIP_CALL( SCIPgetConsExprExprNVars(scip, conshdlr, expr, &(*nlhdlrexprdata)->nvarexprs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*nlhdlrexprdata)->varexprs, (*nlhdlrexprdata)->nvarexprs) );
   SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, expr, (*nlhdlrexprdata)->varexprs, &(*nlhdlrexprdata)->nvarexprs) );

   /* check if expr is semicontinuous and save indicator variables */
   SCIP_CALL( exprIsSemicontinuous(scip, conshdlr, nlhdlrdata, *nlhdlrexprdata, expr, success) );

   if( *success )
   {
      int sindicators;

      sindicators = (*nlhdlrexprdata)->nindicators;

      /* compute 'off' values and propagate semicontinuity */
      SCIP_CALL( computeOffValues(scip, conshdlr, nlhdlrdata, *nlhdlrexprdata, expr) );

      /* some indicator variables might have been removed if evaluation failed, check how many remain */
      if( (*nlhdlrexprdata)->nindicators == 0 )
      {
         *success = FALSE;
      }
      else if( (*nlhdlrexprdata)->nindicators < sindicators )
      {
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(*nlhdlrexprdata)->indicators, sindicators,
               (*nlhdlrexprdata)->nindicators) );
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(*nlhdlrexprdata)->exprvals0, sindicators,
                 (*nlhdlrexprdata)->nindicators) );
      }
   }

   if( *success )
   {
      assert((*nlhdlrexprdata)->nindicators > 0);

#ifdef SCIP_DEBUG
      SCIPinfoMessage(scip, NULL, "detected an on/off expr: ");
      SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
      SCIPinfoMessage(scip, NULL, "\n");
#endif

      /* if we get here, sepa enforcemethods should have already been set by other handler(s) */
      assert(((*enforcemethods & SCIP_CONSEXPR_EXPRENFO_SEPABELOW) && *enforcedbelow)
         || ((*enforcemethods & SCIP_CONSEXPR_EXPRENFO_SEPAABOVE) && *enforcedabove));

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

/** analyse on/off bounds for: 1) tightening bounds in probing for indicator = 1, 2) fixing indicator / detecting
  * cutoff if one or both states is infeasible 3) tightening local bounds if indicator is fixed */
static
SCIP_RETCODE applyOnoffBounds(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HASHMAP*         scvars,             /**< hashmap linking variables to scvardata's */
   SCIP_VAR*             var,                /**< variable */
   SCIP_VAR*             indicator,          /**< indicator variable */
   SCIP_Bool             indvalue,           /**< indicator value for which the bounds are applied */
   SCIP_Bool*            infeas,             /**< pointer to store whether infeasibility has been detected */
   SCIP_Bool*            changed             /**< pointer to store whether bounds have been changed for indvalue = 1 */
   )
{
   SCVARDATA* scvdata;
   int pos;
   SCIP_Bool fixed;
   SCIP_Real sclb;
   SCIP_Real scub;
   SCIP_Bool indfixed;

   *infeas = FALSE;

   /* nothing to do if indicator is already fixed to !indvalue */
   if( (indvalue && SCIPvarGetUbLocal(indicator) == 0) || (!indvalue && SCIPvarGetLbLocal(indicator) == 1))
      return SCIP_OKAY;

   scvdata = getSCVarDataInd(scvars, var, indicator, &pos);

   /* nothing to do for non-semicontinuous variables */
   if( scvdata == NULL )
      return SCIP_OKAY;

   sclb = indvalue ? scvdata->lbs1[pos] : scvdata->vals0[pos];
   scub = indvalue ? scvdata->ubs1[pos] : scvdata->vals0[pos];

   indfixed = SCIPvarGetUbLocal(var) == 0 || SCIPvarGetLbLocal(var) == 1;

   /* use a non-redundant lower bound */
   if( SCIPisGT(scip, sclb, SCIPvarGetLbLocal(var)) )
   {
      if( indvalue )
         *changed = TRUE;

      /* first check for infeasibility */
      if( SCIPisFeasGT(scip, sclb, SCIPvarGetUbLocal(var)) )
      {
         SCIP_CALL( SCIPfixVar(scip, indicator, !indvalue, infeas, &fixed) );
         if( *infeas )
            return SCIP_OKAY;
      }
      else
      { /* sclb is feasible wrt local ub, apply it */
         if( indfixed )
         { /* if indicator is fixed to indvalue, sclb is valid for the current node */
            SCIP_CALL( SCIPchgVarLb(scip, var, sclb) );
         }
         else if( indvalue )
         { /* else we can only fix in probing; we do probing only for indicator = 1 */
            SCIP_CALL( SCIPchgVarLbProbing(scip, var, sclb) );
         }
      }
   }

   /* use a non-redundant upper bound */
   if( SCIPisLT(scip, scub, SCIPvarGetUbLocal(var)) )
   {
      if( indvalue )
         *changed = TRUE;

      /* first check for infeasibility */
      if( SCIPisFeasLT(scip, scub, SCIPvarGetLbLocal(var)) )
      {
         SCIP_CALL( SCIPfixVar(scip, indicator, 0, infeas, &fixed) );
         if( *infeas )
            return SCIP_OKAY;
      }
      else
      { /* scub is feasible wrt local ub, apply it */
         if( indfixed )
         { /* if indicator is fixed to indvalue, scub is valid for the current node */
            SCIP_CALL( SCIPchgVarUb(scip, var, scub) );
         }
         else if( indvalue )
         { /* else we can only fix in probing; we do probing only for indicator = 1 */
            SCIP_CALL( SCIPchgVarUbProbing(scip, var, scub) );
         }
      }
   }

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "%s in [%g, %g] instead of [%g, %g] (vals0 = %g)\n", SCIPvarGetName(var), SCIPvarGetLbLocal(var),
                            SCIPvarGetUbLocal(var), oldlb, oldub, scvdata->vals0[pos]);
#endif

   return SCIP_OKAY;
}

/** nonlinear handler enforcement callback
 *
 * "Perspectivies" cuts produced by other handlers. Suppose that we want to separate x from the set g(x) <= 0.
 * If g(x) = g0 if indicator z = 0, and a cut is given by sum aixi + c <= aux, where xi = xi0 if z = 0 for all i,
 * then the "perspectivied" cut is sum aixi + c + (1 - z)*(g0 - c - sum aix0i) <= aux. This ensures that at z = 1,
 * the new cut is equivalent to the given cut, and at z = 0 it reduces to g0 <= aux.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRENFO(nlhdlrEnfoPerspective)
{ /*lint --e{715}*/
   SCIP_ROWPREP* rowprep;
   SCIP_VAR* auxvar;
   int i;
   int j;
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;
   SCIP_Real cst0;
   SCIP_VAR* indicator;
   SCIP_PTRARRAY* rowpreps2;
   SCIP_PTRARRAY* rowpreps;
   int nrowpreps;
   SCIP_Bool addedbranchscores = FALSE;

   nlhdlrdata = SCIPgetConsExprNlhdlrData(nlhdlr);

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "enforcement method of perspective nonlinear handler called for expr %p: ", expr);
   SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, expr, NULL) );
   SCIPinfoMessage(scip, NULL, "\n");
#endif

   assert(scip != NULL);
   assert(expr != NULL);
   assert(conshdlr != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(nlhdlrdata != NULL);

   auxvar = SCIPgetConsExprExprAuxVar(expr);
   assert(auxvar != NULL);

   nrowpreps = 0;
   *result = SCIP_DIDNOTFIND;

   SCIP_CALL( SCIPcreatePtrarray(scip, &rowpreps2) );
   SCIP_CALL( SCIPcreatePtrarray(scip, &rowpreps) );

   /* build cuts for every indicator variable */
   for( i = 0; i < nlhdlrexprdata->nindicators; ++i )
   {
      int v;
      int minidx;
      int maxidx;
      int r;
      SCIP_Bool changed;
      SCIP_Bool cutoff;
      SCIP_Bool var_is_sc;
      SCIP_Real val0;
      SCIP_Longint ndomreds;

      indicator = nlhdlrexprdata->indicators[i];

      /* go into probing */
      SCIP_CALL( SCIPstartProbing(scip) );

      /* create a probing node */
      SCIP_CALL( SCIPnewProbingNode(scip) );

      changed = FALSE;

      /* change bounds to those at indicators[i] = 1 */
      for( v = 0; v < nlhdlrexprdata->nvarexprs; ++v )
      {
         SCIP_VAR* var;
         SCIP_Bool infeas;
         int b;
#ifdef SCIP_DEBUG
         SCIP_Real oldlb;
         SCIP_Real oldub;
#endif

         var = SCIPgetConsExprExprVarVar(nlhdlrexprdata->varexprs[v]);

#ifdef SCIP_DEBUG
         oldlb = SCIPvarGetLbLocal(var);
         oldub = SCIPvarGetUbLocal(var);
#endif

         for( b = 0; b < 2; ++b )
         {
            SCIP_Bool indvalue = b == 1;

            SCIP_CALL( applyOnoffBounds(scip, nlhdlrdata->scvars, var, indicator, indvalue, &infeas, &changed) );

            if( infeas )
            {
               *result = SCIP_CUTOFF;
               SCIPendProbing(scip);
               goto TERMINATE;
            }
         }
      }

      /* don't add cuts for fixed indicators since there is no use for perspectivy */
      if( SCIPvarGetLbLocal(indicator) == 1 )
      {
         SCIPendProbing(scip);
         continue;
      }
      if( SCIPvarGetUbLocal(indicator) == 0 )
      { /* this case is stronger as it implies that everything is fixed;
         * therefore we are now happy
         */
         int nchildren;
         int pos;
         SCIP_CONSEXPR_EXPR** children;
         SCVARDATA* scvdata;

         SCIPendProbing(scip);

         /* the cut is trivial, can add it without using other nlhdlrs */

         /* vars should have been fixed by now, so just pass NULL as solution (TODO check if this works indeed) */
         SCIP_CALL( SCIPevalConsExprExpr(scip, conshdlr, expr, NULL, 0) );

         nchildren = SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrSum(conshdlr) ?
               SCIPgetConsExprExprNChildren(expr) : 1;

         SCIP_CALL( SCIPcreateRowprep(scip, &rowprep, overestimate ? SCIP_SIDETYPE_LEFT : SCIP_SIDETYPE_RIGHT, TRUE) );

         if( nchildren == 1 )
         {
            SCIPaddRowprepConstant(rowprep, SCIPgetConsExprExprValue(expr));
         }
         else
         {
            for( j = 0; j < nchildren; ++j )
            {
               SCIP_CONSEXPR_EXPR* child = SCIPgetConsExprExprChildren(expr)[j];
               SCIP_VAR* childauxvar = SCIPgetConsExprExprAuxVar(child);

               if( childauxvar == NULL )
                  SCIPaddRowprepConstant(rowprep, SCIPgetConsExprExprValue(child));
               else
               {
                  SCIP_CALL(varIsSemicontinuous(scip, childauxvar, nlhdlrdata->scvars, indicator, &val0, &var_is_sc));
                  scvdata = getSCVarDataInd(nlhdlrdata->scvars, childauxvar, indicator, &pos);

                  if( scvdata != NULL )
                     SCIPaddRowprepConstant(rowprep, SCIPgetConsExprExprValue(child));
                  else
                     SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, childauxvar, SCIPgetConsExprExprSumCoefs(child)[j]) );
               }
            }
         }

         SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, auxvar, -1.0) );
         SCIP_CALL( addCut(scip, cons, rowprep, sol, result) );
         SCIPfreeRowprep(scip, &rowprep);

         goto TERMINATE;
      }

      if( changed && SCIPgetDepth(scip) == 0 )
      {
         SCIP_CALL( SCIPpropagateProbing(scip, nlhdlrdata->maxproprounds, &cutoff, &ndomreds) );
         SCIPdebugMsg(scip, "ndomreds = %d\n", ndomreds);
      }

      /* use cuts from every suitable nlhdlr */
      for( j = 0; j < expr->nenfos; ++j )
      {
         SCIP_Bool addedbranchscores2;
         SCIP_CONSEXPR_NLHDLR* nlhdlr2;
         SCIP_Bool success2;

         nlhdlr2 = expr->enfos[j]->nlhdlr;

         if( !SCIPhasConsExprNlhdlrEstimate(nlhdlr2) || nlhdlr2 == nlhdlr )
            continue;

         SCIPdebugMsg(scip, "asking nonlinear handler %s to %sestimate\n", SCIPgetConsExprNlhdlrName(nlhdlr2), overestimate ? "over" : "under");

         /* evaluate auxiliary before calling estimate */
         SCIP_CALL( SCIPevalauxConsExprNlhdlr(scip, nlhdlr2, expr, expr->enfos[j]->nlhdlrexprdata, &expr->enfos[j]->auxvalue, sol) );

         /* ask the nonlinear handler for an estimator */
         SCIP_CALL( SCIPestimateConsExprNlhdlr(scip, conshdlr, nlhdlr2, expr, expr->enfos[j]->nlhdlrexprdata, sol, expr->enfos[j]->auxvalue,
               overestimate, SCIPgetSolVal(scip, sol, auxvar), rowpreps2, &success2, FALSE, &addedbranchscores2) );

         addedbranchscores += addedbranchscores2;

         minidx = SCIPgetPtrarrayMinIdx(scip, rowpreps2);
         maxidx = SCIPgetPtrarrayMaxIdx(scip, rowpreps2);

         assert((success2 && minidx <= maxidx) || (!success2 && minidx > maxidx));

         for( r = minidx; r <= maxidx; ++r )
         {
            SCIP_VAR *var;

            rowprep = (SCIP_ROWPREP*) SCIPgetPtrarrayVal(scip, rowpreps2, r);

            assert(rowprep != NULL);

#ifdef SCIP_DEBUG
            SCIPdebugMsg(scip, "rowprep for expr ");
            SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
            SCIPinfoMessage(scip, NULL, " before perspectivy is: \n");
            SCIPprintRowprep(scip, rowprep, NULL);
#endif

            /* perspectivy the estimator by adding (1-z)(g0 - c - sum aix0i),
             * where sum aixi + c = rowprep */

            /* we want cst0 = g0 - c - sum aix0i; first add g0 - c */
            cst0 = nlhdlrexprdata->exprvals0[i] + rowprep->side;

            for( v = 0; v < rowprep->nvars; ++v )
            {
               var = rowprep->vars[v];

               /* is var sc with respect to this indicator? */
               SCIP_CALL(varIsSemicontinuous(scip, var, nlhdlrdata->scvars, indicator, &val0, &var_is_sc));

               if( !var_is_sc )
                  continue;

               cst0 -= rowprep->coefs[v] * val0;
            }

            /* update the rowprep by adding cst0 - cst0*z */
            SCIPaddRowprepConstant(rowprep, cst0);
            SCIP_CALL(SCIPaddRowprepTerm(scip, rowprep, indicator, -cst0));

            SCIP_CALL(SCIPaddRowprepTerm(scip, rowprep, auxvar, -1.0));

            SCIPdebugMsg(scip, "rowprep after perspectivy is: \n");
#ifdef SCIP_DEBUG
            SCIPprintRowprep(scip, rowprep, NULL);
#endif

            SCIP_CALL( SCIPsetPtrarrayVal(scip, rowpreps, nrowpreps, rowprep) );
            ++nrowpreps;
         }

         SCIP_CALL( SCIPclearPtrarray(scip, rowpreps2) );
      }

      SCIP_CALL( SCIPendProbing(scip) );

      /* add the cuts */
      minidx = SCIPgetPtrarrayMinIdx(scip, rowpreps);
      maxidx = SCIPgetPtrarrayMaxIdx(scip, rowpreps);

      for( r = minidx; r <= maxidx; ++r )
      {
#ifdef SCIP_DEBUG
         SCIPprintRowprep(scip, rowprep, NULL);
#endif
         rowprep = (SCIP_ROWPREP*) SCIPgetPtrarrayVal(scip, rowpreps, r);

         SCIP_CALL( SCIPconsExprCutAndScore(scip, conshdlr, nlhdlr, cons, expr, rowprep, overestimate, auxvar,
               auxvalue, allowweakcuts, addedbranchscores, addbranchscores, sol, result) );
         SCIPfreeRowprep(scip, &rowprep);
      }
      SCIP_CALL( SCIPclearPtrarray(scip, rowpreps) );
   }

TERMINATE:
   SCIP_CALL( SCIPfreePtrarray(scip, &rowpreps2) );
   SCIP_CALL( SCIPfreePtrarray(scip, &rowpreps) );

   return SCIP_OKAY;
}

/** nonlinear handler estimation callback
 *
 * "Perspectivies" cuts produced by other handlers. Suppose that we want to separate x from the set g(x) <= 0.
 * If g(x) = g0 if indicator z = 0, and a cut is given by sum aixi + c <= aux, where xi = xi0 if z = 0 for all i,
 * then the "perspectivied" cut is sum aixi + c + (1 - z)*(g0 - c - sum aix0i) <= aux. This ensures that at z = 1,
 * the new cut is equivalent to the given cut, and at z = 0 it reduces to g0 <= aux.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRESTIMATE(nlhdlrEstimatePerspective)
{ /*lint --e{715}*/
   SCIP_ROWPREP* rowprep;
   SCIP_VAR* auxvar;
   int i;
   int j;
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;
   SCIP_Real cst0;
   SCIP_VAR* indicator;
   SCIP_PTRARRAY* rowpreps2;
   int nrowpreps;

   nlhdlrdata = SCIPgetConsExprNlhdlrData(nlhdlr);

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "enforcement method of perspective nonlinear handler called for expr %p: ", expr);
   SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, expr, NULL) );
   SCIPinfoMessage(scip, NULL, "\n");
#endif

   assert(scip != NULL);
   assert(expr != NULL);
   assert(conshdlr != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(rowpreps != NULL);
   assert(nlhdlrdata != NULL);

   auxvar = SCIPgetConsExprExprAuxVar(expr);
   assert(auxvar != NULL);

   nrowpreps = 0;
   *success = FALSE;

   SCIP_CALL( SCIPcreatePtrarray(scip, &rowpreps2) );

   /* build cuts for every indicator variable */
   for( i = 0; i < nlhdlrexprdata->nindicators; ++i )
   {
      int v;
      int pos;
      SCIP_Bool changed;
      SCIP_Bool cutoff;
      SCIP_Longint ndomreds;

      indicator = nlhdlrexprdata->indicators[i];

      if( SCIPvarGetLbLocal(indicator) == 1 || SCIPvarGetUbLocal(indicator) == 0 )
      {
         continue; /* nothing to do if indicator is already fixed */
      }

      /* go into probing */
      SCIP_CALL( SCIPstartProbing(scip) );

      /* create a probing node */
      SCIP_CALL( SCIPnewProbingNode(scip) );

      changed = FALSE;

      SCIPfixVarProbing(scip, indicator, 1);

      /* change bounds to those at indicators[i] = 1 */
      for( v = 0; v < nlhdlrexprdata->nvarexprs; ++v )
      {
         SCIP_VAR* var;
         SCVARDATA* scvdata;
#ifdef SCIP_DEBUG
         SCIP_Real oldlb;
         SCIP_Real oldub;
#endif

         var = SCIPgetConsExprExprVarVar(nlhdlrexprdata->varexprs[v]);
         scvdata = getSCVarDataInd(nlhdlrdata->scvars, var, indicator, &pos);
         assert(scvdata != NULL);

#ifdef SCIP_DEBUG
         oldlb = SCIPvarGetLbLocal(var);
         oldub = SCIPvarGetUbLocal(var);
#endif

         if( SCIPisGT(scip, scvdata->lbs1[pos], SCIPvarGetLbLocal(var)) )
         {
              changed = TRUE;
              SCIP_CALL( SCIPchgVarLbProbing(scip, var, scvdata->lbs1[pos]) );
         }

         if( SCIPisLT(scip, scvdata->ubs1[pos], SCIPvarGetUbLocal(var)) )
         {
             changed = TRUE;
             SCIP_CALL( SCIPchgVarUbProbing(scip, var, scvdata->ubs1[pos]) );
         }

#ifdef SCIP_DEBUG
         SCIPdebugMsg(scip, "%s in [%g, %g] instead of [%g, %g] (vals0 = %g)\n", SCIPvarGetName(var), SCIPvarGetLbLocal(var),
                            SCIPvarGetUbLocal(var), oldlb, oldub, scvdata->vals0[pos]);
#endif
      }

      if( changed && SCIPgetDepth(scip) == 0 )
      {
         SCIP_CALL( SCIPpropagateProbing(scip, nlhdlrdata->maxproprounds, &cutoff, &ndomreds) );
         SCIPdebugMsg(scip, "ndomreds = %d\n", ndomreds);
      }

      /* use cuts from every suitable nlhdlr */
      for( j = 0; j < expr->nenfos; ++j )
      {
         SCIP_Bool addedbranchscores2;
         SCIP_CONSEXPR_NLHDLR* nlhdlr2;
         int minidx;
         int maxidx;
         int r;
         SCIP_Bool success2;

         nlhdlr2 = expr->enfos[j]->nlhdlr;

         if( !SCIPhasConsExprNlhdlrEstimate(nlhdlr2) || nlhdlr2 == nlhdlr )
            continue;

         SCIPdebugMsg(scip, "asking nonlinear handler %s to %sestimate\n", SCIPgetConsExprNlhdlrName(nlhdlr2), overestimate ? "over" : "under");

         /* evaluate auxiliary before calling estimate */
         SCIP_CALL( SCIPevalauxConsExprNlhdlr(scip, nlhdlr2, expr, expr->enfos[j]->nlhdlrexprdata, &expr->enfos[j]->auxvalue, sol) );

         /* ask the nonlinear handler for an estimator */
         SCIP_CALL( SCIPestimateConsExprNlhdlr(scip, conshdlr, nlhdlr2, expr, expr->enfos[j]->nlhdlrexprdata, sol, expr->enfos[j]->auxvalue,
               overestimate, SCIPgetSolVal(scip, sol, auxvar), rowpreps2, &success2, FALSE, &addedbranchscores2) );

         minidx = SCIPgetPtrarrayMinIdx(scip, rowpreps2);
         maxidx = SCIPgetPtrarrayMaxIdx(scip, rowpreps2);

         assert((success2 && minidx <= maxidx) || (!success2 && minidx > maxidx));

         for( r = minidx; r <= maxidx; ++r )
         {
            SCIP_VAR *var;
            SCIP_Bool var_is_sc;

            rowprep = (SCIP_ROWPREP*) SCIPgetPtrarrayVal(scip, rowpreps2, r);

            assert(rowprep != NULL);

#ifdef SCIP_DEBUG
            SCIPdebugMsg(scip, "rowprep for expr ");
            SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
            SCIPinfoMessage(scip, NULL, " before perspectivy is: \n");
            SCIPprintRowprep(scip, rowprep, NULL);
#endif

            /* perspectivy the estimator by adding (1-z)(g0 - c - sum aix0i),
             * where sum aixi + c = rowprep */

            /* we want cst0 = g0 - c - sum aix0i; first add g0 - c */
            cst0 = nlhdlrexprdata->exprvals0[i] + rowprep->side;

            for( v = 0; v < rowprep->nvars; ++v )
            {
               SCIP_Real val0;

               var = rowprep->vars[v];

               /* is var sc with respect to this indicator? */
               SCIP_CALL(varIsSemicontinuous(scip, var, nlhdlrdata->scvars, indicator, &val0, &var_is_sc));

               assert(var_is_sc);

               cst0 -= rowprep->coefs[v] * val0;
            }

            /* update the rowprep by adding cst0 - cst0*z */
            SCIPaddRowprepConstant(rowprep, cst0);
            SCIP_CALL(SCIPaddRowprepTerm(scip, rowprep, indicator, -cst0));

            SCIPdebugMsg(scip, "rowprep after perspectivy is: \n");
#ifdef SCIP_DEBUG
            SCIPprintRowprep(scip, rowprep, NULL);
#endif

            *success = TRUE;
            SCIP_CALL( SCIPsetPtrarrayVal(scip, rowpreps, nrowpreps, rowprep) );
            ++nrowpreps;
         }

         SCIP_CALL( SCIPclearPtrarray(scip, rowpreps2) );
      }

      SCIP_CALL( SCIPendProbing(scip) );
   }

   SCIP_CALL( SCIPfreePtrarray(scip, &rowpreps2) );

   return SCIP_OKAY;
}

/** find the intersection of two disconnected domains given as arrays of intervals */
static
void intersectDisconnectedDomains(
   SCIP_INTERVAL*        intervals1,
   SCIP_INTERVAL*        intervals2,
   SCIP_INTERVAL*        intervals,
   int                   n1,
   int                   n2,
   int*                  n
   )
{
   int i1;
   int i2;
   SCIP_INTERVAL interval;

   assert(intervals1 != NULL);
   assert(intervals2 != NULL);
   assert(intervals != NULL);
   assert(n != NULL);

   i1 = 0;
   i2 = 0;
   *n = 0;

   while( i1 < n1 && i2 < n2 )
   {
      SCIPintervalIntersect(&interval, intervals1[i1], intervals2[i2]);

      /* if the result is empty, choose the array with the leftmost current interval
       * and shift to the next interval there
       */
      if( interval.inf > interval.sup )
      {
         if( intervals1[i1].inf < intervals2[i2].inf )
            ++i1;
         else
            ++i2;
         continue;
      }

      /* if we've reached the end of an interval, switch to the next one */
      if( intervals1[i1].sup == interval.sup )
         ++i1;
      if( intervals2[i2].sup == interval.sup )
         ++i2;

      intervals[*n] = interval;
      ++(*n);
   }
}

static
void printDisconnectedDomain(
   SCIP*                 scip,
   SCIP_INTERVAL*        intervals,
   int                   n
   )
{
   int i;

   SCIPinfoMessage(scip, NULL, "\ndomain: ");
   for( i = 0; i < n; ++i )
   {
      SCIPinfoMessage(scip, NULL, "[%g, %g]; ", intervals[i].inf, intervals[i].sup);
   }
}

/** nonlinear handler interval evaluation callback
 *
 *  Fixes each indicator to either 0 or 1 and propagates the change.
 *  If one of the states (on or off) is infeasible, fixes the indicator to the other.
 *  Uses information from all indicators to update the range of the expression.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRINTEVAL(nlhdlrIntevalPerspective)
{ /*lint --e{715}*/
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;
   int i;
   int ncur;
   int nres;
   int sres;
   int scur;
   SCIP_VAR* auxvar;
   SCVARDATA* wscvdata;
   SCIP_INTERVAL* curintervals;
   SCIP_INTERVAL* resintervals;
   SCIP_CONSHDLR* conshdlr;
   SCIP_Bool success;

   if( SCIPinProbing(scip) || SCIPinRepropagation(scip) )
      return SCIP_OKAY;

   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(expr != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(interval != NULL);

   nlhdlrdata = SCIPgetConsExprNlhdlrData(nlhdlr);
   auxvar = SCIPgetConsExprExprAuxVar(expr);
   wscvdata = (SCVARDATA*) SCIPhashmapGetImage(nlhdlrdata->scvars, (void*)auxvar);
   conshdlr = SCIPfindConshdlr(scip, "expr");

   assert(conshdlr != NULL);
   assert(nlhdlrdata != NULL);
   assert(auxvar != NULL);
   assert(wscvdata != NULL);

   ncur = 1;
   scur = 1;
   nres = 2;
   sres = 2;
   SCIP_CALL( SCIPallocBufferArray(scip, &curintervals, scur) );
   SCIP_CALL( SCIPallocBufferArray(scip, &resintervals, sres) );

   /* set curintervals to the current activity interval */
   curintervals[0] = *interval;
   success = FALSE;

   SCIPdebugMsg(scip, "perspective inteval called with interval = [%g, %g]\n", SCIPintervalGetInf(*interval),
                SCIPintervalGetSup(*interval));

   for( i = 0; i < nlhdlrexprdata->nindicators; ++i )
   {
      SCIP_INTERVAL indintervals[2];
      int nind;
      int snew;
      int v;
      int pos;
      int posw;
      SCIP_VAR* indicator;
      SCIP_VAR* var;
      SCIP_Bool fixed;
      SCIP_Bool cutoff;
      SCIP_Bool changed;
      SCIP_Bool infeasible;
      SCVARDATA* scvdata;

      indicator = nlhdlrexprdata->indicators[i];

      /* nothing to do if the indicator is already fixed */
      if( SCIPvarGetLbLocal(indicator) == 1 || SCIPvarGetUbLocal(indicator) == 0 )
         continue;

      /* TODO probing at 0 too */
      /* TODO make sure to avoid infinite probing */

      changed = FALSE;

      /* see if we can tighten the bounds for at least one var by setting indicator = 1 */
      for( v = 0; v < nlhdlrexprdata->nvarexprs; ++v )
      {
         var = SCIPgetConsExprExprVarVar(nlhdlrexprdata->varexprs[v]);
         scvdata = getSCVarDataInd(nlhdlrdata->scvars, var, indicator, &pos);
         assert(scvdata != NULL);

         if( SCIPisGT(scip, scvdata->lbs1[pos], SCIPvarGetLbLocal(var)) ||
             SCIPisLT(scip, scvdata->ubs1[pos], SCIPvarGetUbLocal(var)) )
         {
            changed = TRUE;
            break;
         }
      }

      if( changed )
      {
         SCIPdebugMsg(scip, "fixing %s = 1\n", SCIPvarGetName(indicator));

         /* go into probing */
         SCIP_CALL( SCIPstartProbing(scip) );

         SCIP_CALL( SCIPnewProbingNode(scip) );
         SCIP_CALL( SCIPfixVarProbing(scip, indicator, 1) );

         SCIP_CALL( SCIPpropagateProbing(scip, nlhdlrdata->maxproprounds, &cutoff, NULL) );

         /* TODO note: propagate probing might detect more proper sc expressions */

         if( cutoff )
         {
            SCIPdebugMsg(scip, "cutoff!\n");
            /* if this fixing is infeasible, fix indicator to the other value */
            SCIP_CALL( SCIPfixVar(scip, indicator, 0, &infeasible, &fixed) );

            if( infeasible )
            {
               /* if the other fixing is also infeasible, the node is infeasible */
               SCIP_CALL( SCIPendProbing(scip) );
               SCIPintervalSetEmpty(interval);
               goto TERMINATE;
            }

            /* indicator is fixed -> w0 cup [wlb1,wub1] = w0 */
            SCIPevalConsExprExprActivity(scip, conshdlr, expr, &indintervals[0], FALSE);

            /* TODO as long as we only allow expressions where all vars are semicontinuous, this should hold */
            assert(indintervals[0].inf == indintervals[0].sup);

            nind = 1;
         }
         else
         {
            SCIPdebugMsg(scip, "no cutoff\n");

            SCIPsortedvecFindPtr((void**)wscvdata->bvars, SCIPvarComp, (void*)indicator, wscvdata->nbnds, &posw);

            /* save w0 cup [wlb1,wub1] */
            indintervals[0].inf = MIN(wscvdata->vals0[posw], wscvdata->lbs1[posw]);

            if( wscvdata->lbs1[posw] <= wscvdata->vals0[posw] && wscvdata->vals0[posw] <= wscvdata->ubs1[posw] )
            {
               /* val0 in [lb1,ub1] -> we have only one interval */
               indintervals[0].sup = MAX(wscvdata->vals0[posw], wscvdata->ubs1[posw]);
               SCIPintervalSetEmpty(&indintervals[1]);
               nind = 1;
            }
            else
            {
               indintervals[0].sup = MIN(wscvdata->vals0[posw], wscvdata->ubs1[posw]);
               indintervals[1].inf = MAX(wscvdata->vals0[posw], wscvdata->lbs1[posw]);
               indintervals[1].sup = MAX(wscvdata->vals0[posw], wscvdata->ubs1[posw]);
               nind = 2;
            }
         }

#ifdef SCIP_DEBUG
         SCIPdebugMsg(scip, "indintervals: ");
         printDisconnectedDomain(scip, indintervals, nind);
         SCIPdebugMsg(scip, "curintervals: ");
         printDisconnectedDomain(scip, curintervals, ncur);
#endif

         /* intersect curinterval with intervali = w0 cup [wlb1,wub1] */
         snew = ncur == nind ? ncur * 2 - 1 : MIN(ncur, nind) * 2;
         if( snew > sres )
         {
            sres = snew;
            SCIP_CALL( SCIPreallocBufferArray(scip, &resintervals, sres) );
         }
         intersectDisconnectedDomains(curintervals, indintervals, resintervals, ncur, nind, &nres);

#ifdef SCIP_DEBUG
         SCIPdebugMsg(scip, "resintervals: ");
         printDisconnectedDomain(scip, resintervals, nres);
#endif

         if( i < nlhdlrexprdata->nindicators - 1 )
         {
            /* save the result to curintervals */
            if( sres > scur )
            {
               scur = sres;
               SCIP_CALL( SCIPreallocBufferArray(scip, &curintervals, scur) );
            }
            BMScopyMemoryArray(curintervals, resintervals, nres);
            ncur = nres;
         }

         /* TODO handle cutoff */

         SCIP_CALL( SCIPendProbing(scip) );

         success = TRUE;
      }
   }

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "final resintervals: ");
   printDisconnectedDomain(scip, resintervals, nres);
#endif

   if( success )
      SCIPintervalSetBounds(interval, resintervals[0].inf, resintervals[nres - 1].sup);

 TERMINATE:
   SCIPfreeBufferArray(scip, &resintervals);
   SCIPfreeBufferArray(scip, &curintervals);

   return SCIP_OKAY;
}

/** nonlinear handler callback for reformulation */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRREFORMULATE(nlhdlrReformulatePerspective)
{ /*lint --e{715}*/

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

   SCIP_CALL( SCIPincludeConsExprNlhdlrBasic(scip, consexprhdlr, &nlhdlr, NLHDLR_NAME, NLHDLR_DESC, NLHDLR_DETECTPRIORITY,
      NLHDLR_ENFOPRIORITY, nlhdlrDetectPerspective, nlhdlrEvalauxPerspective, nlhdlrdata) );
   assert(nlhdlr != NULL);

   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/expr/nlhdlr/" NLHDLR_NAME "/detectsum",
      "whether to run convexity detection when the root of an expression is a sum",
      &nlhdlrdata->detectsum, FALSE, DEFAULT_DETECTSUM, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "constraints/expr/nlhdlr/" NLHDLR_NAME "/multcuts",
      "whether to add cuts for all suitable indicator variables",
      &nlhdlrdata->multcuts, FALSE, DEFAULT_MULTCUTS, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "constraints/expr/nlhdlr/" NLHDLR_NAME "/maxproprounds",
           "maximal number of propagation rounds in probing",
           &nlhdlrdata->maxproprounds, FALSE, DEFAULT_MAXPROPROUNDS, -1, INT_MAX, NULL, NULL) );

   SCIPsetConsExprNlhdlrCopyHdlr(scip, nlhdlr, nlhdlrCopyhdlrPerspective);
   SCIPsetConsExprNlhdlrFreeHdlrData(scip, nlhdlr, nlhdlrFreehdlrdataPerspective);
   SCIPsetConsExprNlhdlrFreeExprData(scip, nlhdlr, nlhdlrFreeExprDataPerspective);
   /* SCIPsetConsExprNlhdlrSepa(scip, nlhdlr, NULL, NULL, nlhdlrEstimatePerspective, NULL); */
   SCIPsetConsExprNlhdlrSepa(scip, nlhdlr, NULL, nlhdlrEnfoPerspective, NULL, NULL);

   return SCIP_OKAY;
}
