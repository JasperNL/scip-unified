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

/**@file   cons_expr_nlhdlr_perspective.h
 * @brief  perspective nonlinear handler
 * @author Benjamin Mueller
 */

#include <string.h>

#include "scip/cons_varbound.h"
#include "scip/cons_expr_nlhdlr_perspective.h"
#include "scip/cons_expr.h"
#include "scip/cons_expr_var.h"
#include "scip/cons_expr_sum.h"
#include "scip/type_cons_expr.h"
#include "scip/struct_cons_expr.h"
#include "scip/scip_sol.h"


/* fundamental nonlinear handler properties */
#define NLHDLR_NAME         "perspective"
#define NLHDLR_DESC         "perspective handler for expressions"
#define NLHDLR_PRIORITY     150

/*
 * Data structures
 */

/** nonlinear handler expression data */
struct SCIP_ConsExpr_NlhdlrExprData
{
   SCIP_EXPRCURV curvature;        /**< curvature of the on/off part of the expression */
   SCIP_HASHMAP* scvars;           /**< maps semicontinuous variables to their on/off bounds */
   SCIP_HASHMAP* persp;            /**< maps binary variables to on/off expressions */
   SCIP_CONSEXPR_EXPR** linterms;  /**< terms for which we do not apply perspective cuts */
   SCIP_Real* lincoefs;            /**< coefficients of linterms */
   int nperspterms;                /**< number of on/off expressions */
   int nlinterms;                  /**< number of linterms */
   int lintermssize;               /**< size of the linterms array */
};

/** nonlinear handler data */
struct SCIP_ConsExpr_NlhdlrData
{
};

/*
 * Local methods
 */

/* a variable is semicontinuous if its bounds depend on the binary variable bvar
 * and bvar == 0 => var = v_off for some real constant v_off.
 * If the bvar is not specified, find the first binary variable that var depends on.
 */
static
SCIP_Bool varIsSemicontinuous(
   SCIP*          scip,       /**< SCIP data structure */
   SCIP_VAR*      var,        /**< the variable to check */
   SCIP_VAR**     bvar,       /**< the binary variable */
   SCIP_HASHMAP*  scvars      /**< semicontinuous variable information */
   )
{
   SCIP_SCVARDATA* scv;
   SCIP_CONSHDLR* varboundconshdlr;
   SCIP_CONS** vbconss;
   SCIP_SCVARDATA* olddata;
   int nvbconss, c;
   SCIP_Real lb0, ub0, lb1, ub1;

   assert(scip != NULL);
   assert(var != NULL);

   olddata = SCIPhashmapGetImage(scvars, (void*)var);
   if( olddata != NULL )
   {
      if( *bvar == NULL )
         *bvar = olddata->bvar;
      if( olddata->bvar == *bvar )
         return TRUE;
      else
         return FALSE;
   }

   lb0 = -SCIPinfinity(scip);
   ub0 = SCIPinfinity(scip);
   lb1 = -SCIPinfinity(scip);
   ub1 = SCIPinfinity(scip);

   varboundconshdlr = SCIPfindConshdlr(scip, "varbound");
   nvbconss = SCIPconshdlrGetNConss(varboundconshdlr);
   vbconss = SCIPconshdlrGetConss(varboundconshdlr);

   for( c = 0; c < nvbconss; ++c )
   {
      /* look for varbounds containing var and, if specified, bvar */
      if( SCIPgetVarVarbound(scip, vbconss[c]) != var || (*bvar != NULL && SCIPgetVbdvarVarbound(scip, vbconss[c]) != *bvar) )
         continue;

      if( *bvar == NULL )
         *bvar = SCIPgetVbdvarVarbound(scip, vbconss[c]);
#ifdef SCIP_DEBUG
      SCIP_CALL( SCIPprintCons(scip, vbconss[c], NULL) );
      SCIPdebugMsg(scip, "\n");
#endif

      if( SCIPgetLhsVarbound(scip, vbconss[c]) > lb0 )
         lb0 = SCIPgetLhsVarbound(scip, vbconss[c]);
      if( SCIPgetRhsVarbound(scip, vbconss[c]) < ub0 )
         ub0 = SCIPgetRhsVarbound(scip, vbconss[c]);
      if( SCIPgetLhsVarbound(scip, vbconss[c]) - SCIPgetVbdcoefVarbound(scip, vbconss[c]) > lb1 )
         lb1 = SCIPgetLhsVarbound(scip, vbconss[c]) - SCIPgetVbdcoefVarbound(scip, vbconss[c]);
      if( SCIPgetRhsVarbound(scip, vbconss[c]) - SCIPgetVbdcoefVarbound(scip, vbconss[c]) > ub1 )
         ub1 = SCIPgetRhsVarbound(scip, vbconss[c]) - SCIPgetVbdcoefVarbound(scip, vbconss[c]);
   }

   if( SCIPvarGetLbGlobal(var) >= lb0 )
      lb0 = SCIPvarGetLbGlobal(var);
   if( SCIPvarGetUbGlobal(var) <= ub0 )
      ub0 = SCIPvarGetUbGlobal(var);
   if( SCIPvarGetLbGlobal(var) >= lb1 )
      lb1 = SCIPvarGetLbGlobal(var);
   if( SCIPvarGetUbGlobal(var) <= ub1 )
      ub1 = SCIPvarGetUbGlobal(var);

   SCIPdebugMsg(scip, "\nonoff bounds: lb0 = %f, ub0 = %f, lb1 = %f, ub1 = %f", lb0, ub0, lb1, ub1);

   if( lb0 == ub0 && (lb0 != lb1 || ub0 != ub1) )
   {
      SCIPallocBlockMemory(scip, &scv);
      scv->bvar = *bvar;
      scv->val0 = lb0;
      if( lb0 != 0.0 )
         SCIPdebugMsg(scip, "var %s has a non-zero off value %f", SCIPvarGetName(var), lb0);
      SCIPhashmapInsert(scvars, var, scv);
      return TRUE;
   }

   return FALSE;
}

static
SCIP_RETCODE nlhdlrexprdataAddPerspTerm(
   SCIP*                          scip,            /**< SCIP data structure */
   SCIP_CONSHDLR*                 conshdlr,        /**< constraint handler */
   SCIP_CONSEXPR_NLHDLREXPRDATA*  nlhdlrexprdata,  /**< nonlinear handler expression data */
   SCIP_Real                      coef,            /**< coef of the added term */
   SCIP_CONSEXPR_EXPR*            expr,            /**< expr to add */
   SCIP_VAR*                      bvar             /**< the binary variable */
   )
{
   SCIP_EXPRCURV ecurv;
   SCIP_EXPRCURV pcurv;
   SCIP_CONSEXPR_EXPR* persp;

   assert(scip != NULL);
   assert(nlhdlrexprdata != NULL);

   ecurv = SCIPgetConsExprExprCurvature(expr);
   if( coef < 0 )
   {
      if( ecurv == SCIP_EXPRCURV_CONVEX )
         ecurv = SCIP_EXPRCURV_CONCAVE;
      else if( ecurv == SCIP_EXPRCURV_CONCAVE )
         ecurv = SCIP_EXPRCURV_CONVEX;
   }
   if( SCIPhashmapExists(nlhdlrexprdata->persp, (void*)bvar) )
   {
      persp = SCIPhashmapGetImage(nlhdlrexprdata->persp, (void*)bvar);
      pcurv = SCIPgetConsExprExprCurvature(persp);
      if( pcurv == SCIP_EXPRCURV_LINEAR && (ecurv == SCIP_EXPRCURV_CONVEX || ecurv == SCIP_EXPRCURV_CONCAVE) )
         SCIPsetConsExprExprCurvature(persp, ecurv);
      else if( ecurv == SCIP_EXPRCURV_UNKNOWN || (ecurv != SCIP_EXPRCURV_LINEAR && ecurv != pcurv) )
         SCIPsetConsExprExprCurvature(persp, SCIP_EXPRCURV_UNKNOWN);
      SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, persp, expr, coef) );
   }
   else
   {
      SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, &persp, 1, &expr, &coef, 0.0) );
      SCIPsetConsExprExprCurvature(persp, ecurv);
      SCIP_CALL( SCIPhashmapInsert(nlhdlrexprdata->persp, (void*)bvar, (void*)persp) );
      nlhdlrexprdata->nperspterms++;
   }

   return SCIP_OKAY;
}

static
SCIP_RETCODE nlhdlrexprdataAddLinTerm(
   SCIP*                         scip,
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata,
   SCIP_Real                     coef,
   SCIP_CONSEXPR_EXPR*           expr
   )
{
   int newsize;

   assert(scip != NULL);
   assert(nlhdlrexprdata != NULL);

   if( nlhdlrexprdata->nlinterms + 1 > nlhdlrexprdata->lintermssize )
   {
      newsize = SCIPcalcMemGrowSize(scip, nlhdlrexprdata->nlinterms + 1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->linterms,  nlhdlrexprdata->lintermssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nlhdlrexprdata->lincoefs, nlhdlrexprdata->lintermssize, newsize) );
      nlhdlrexprdata->lintermssize = newsize;
   }
   assert(nlhdlrexprdata->nlinterms + 1 <= nlhdlrexprdata->lintermssize);

   nlhdlrexprdata->lincoefs[nlhdlrexprdata->nlinterms] = coef;
   nlhdlrexprdata->linterms[nlhdlrexprdata->nlinterms] = expr;
   nlhdlrexprdata->nlinterms++;

   return SCIP_OKAY;
}

/** creates auxiliary variable when necessary */
static
SCIP_RETCODE createAuxVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expr conshdlr */
   SCIP_CONSEXPR_EXPR*   expr                /**< expression for which the aux var is created */
)
{
   if( SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrVar(conshdlr) )
      return SCIP_OKAY;

   SCIP_CALL( SCIPcreateConsExprExprAuxVar(scip, conshdlr, expr, NULL) );

   return SCIP_OKAY;
}

/** frees nlhdlrexprdata structure */
static
SCIP_RETCODE freeNlhdlrExprData(
   SCIP*                         scip,            /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata   /**< nlhdlr expression data */
)
{
   SCIP_HASHMAPENTRY* entry;
   SCIP_SCVARDATA* data;
   SCIP_CONSEXPR_EXPR* expr;

   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->linterms), nlhdlrexprdata->lintermssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->lincoefs), nlhdlrexprdata->lintermssize);
   if( nlhdlrexprdata->scvars != NULL )
   {
      for( int c = 0; c < SCIPhashmapGetNEntries(nlhdlrexprdata->scvars); ++c )
      {
         entry = SCIPhashmapGetEntry(nlhdlrexprdata->scvars, c);
         if( entry != NULL )
         {
            data = (SCIP_SCVARDATA*) SCIPhashmapEntryGetImage(entry);
            SCIPfreeBlockMemory(scip, &data);
         }
      }
      SCIPhashmapFree(&(nlhdlrexprdata->scvars));
   }
   if( nlhdlrexprdata->persp != NULL )
   {
      for( int c = 0; c < SCIPhashmapGetNEntries(nlhdlrexprdata->persp); ++c )
      {
         entry = SCIPhashmapGetEntry(nlhdlrexprdata->persp, c);
         if( entry != NULL )
         {
            expr = (SCIP_CONSEXPR_EXPR*) SCIPhashmapEntryGetImage(entry);
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expr) );
         }
      }
      SCIPhashmapFree(&(nlhdlrexprdata->persp));
   }
   return SCIP_OKAY;
}

/*
 * Callback methods of nonlinear handler
 */

/** nonlinear handler copy callback */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRCOPYHDLR(nlhdlrCopyhdlrPerspective)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of perspective nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrCopyhdlrPerspective NULL
#endif

/** callback to free data of handler */
static
SCIP_DECL_CONSEXPR_NLHDLRFREEHDLRDATA(nlhdlrFreehdlrdataPerspective)
{ /*lint --e{715}*/
   SCIPfreeBlockMemory(scip, nlhdlrdata);

   return SCIP_OKAY;
}


/** callback to free expression specific data */
static
SCIP_DECL_CONSEXPR_NLHDLRFREEEXPRDATA(nlhdlrFreeExprDataPerspective)
{  /*lint --e{715}*/
   freeNlhdlrExprData(scip, *nlhdlrexprdata);
   SCIPfreeBlockMemory(scip, nlhdlrexprdata);

   return SCIP_OKAY;
}



/** callback to be called in initialization */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRINIT(nlhdlrInitPerspective)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;
   SCIP_CONSHDLR* linconshdlr;
   SCIP_CONSHDLR* varboundconshdlr;
   int c;

   nlhdlrdata = SCIPgetConsExprNlhdlrData(nlhdlr);
   assert(nlhdlrdata != NULL);

   /* look for important constraint handlers */
   linconshdlr = SCIPfindConshdlr(scip, "linear");
   varboundconshdlr = SCIPfindConshdlr(scip, "varbound");

   for( c = 0; c < SCIPgetNConss(scip); ++c )
   {
      SCIP_CONS* cons = SCIPgetConss(scip)[c];
      assert(cons != NULL);

      if( SCIPconsGetHdlr(cons) == linconshdlr )
      {
         SCIP_CALL( SCIPprintCons(scip, cons, NULL) );
         SCIPinfoMessage(scip, NULL, "\n");

         /* TODO extract information with interface functions from cons_linear.h */
      }
      else if( SCIPconsGetHdlr(cons) == varboundconshdlr )
      {
         SCIP_CALL( SCIPprintCons(scip, cons, NULL) );
         SCIPinfoMessage(scip, NULL, "\n");

         /* TODO extract information with interface functions from cons_varbound.h */
      }
      else
      {
         /* TODO check whether other constraint handlers are important */
      }
   }

   /* TODO store data in nlhdlrdata */

   return SCIP_OKAY;
}
#else
#define nlhdlrInitPerspective NULL
#endif


/** callback to be called in deinitialization */
static
SCIP_DECL_CONSEXPR_NLHDLREXIT(nlhdlrExitPerspective)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}


/** callback to detect structure in expression tree */
static
SCIP_DECL_CONSEXPR_NLHDLRDETECT(nlhdlrDetectPerspective)
{ /*lint --e{715}*/
   SCIP_EXPRCURV child_curvature;
   SCIP_CONSEXPR_EXPR** varexprs;
   SCIP_VAR* var;
   SCIP_VAR* bvar;
   int nvars, v, c, nchildren;
   SCIP_CONSEXPR_EXPR** children;
   SCIP_CONSEXPR_EXPR* pexpr;
   SCIP_Real* coefs;
   SCIP_Bool expr_is_onoff;

   SCIP_CONSEXPR_EXPRENFO_METHOD oldenfo;
   oldenfo = *enforcemethods;

   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(expr != NULL);
   assert(enforcemethods != NULL);
   assert(enforcedbelow != NULL);
   assert(enforcedabove != NULL);
   assert(success != NULL);
   assert(nlhdlrexprdata != NULL);

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "\nCalled perspective detect, expr = ");
   SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
#endif

   *success = FALSE;

   if( *enforcedbelow && *enforcedabove )
      return SCIP_OKAY;

   SCIP_CALL( SCIPallocClearBlockMemory(scip, nlhdlrexprdata) );
   if( !*enforcedbelow && !*enforcedabove )
      (*nlhdlrexprdata)->curvature = SCIP_EXPRCURV_UNKNOWN;
   else if( !*enforcedbelow )
      (*nlhdlrexprdata)->curvature = SCIP_EXPRCURV_CONVEX;
   else if( !*enforcedabove )
      (*nlhdlrexprdata)->curvature = SCIP_EXPRCURV_CONCAVE;

   SCIPgetConsExprExprNVars(scip, conshdlr, expr, &nvars);
   SCIPhashmapCreate(&((*nlhdlrexprdata)->scvars), SCIPblkmem(scip), nvars);

   if( strcmp("sum", SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr))) == 0 )
   {
      children = SCIPgetConsExprExprChildren(expr);
      nchildren = SCIPgetConsExprExprNChildren(expr);
      coefs = SCIPgetConsExprExprSumCoefs(expr);
   }
   else
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &children, 1) );
      *children = expr;
      nchildren = 1;
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &coefs, 1) );
      *coefs = 1.0;
   }
   SCIPhashmapCreate(&((*nlhdlrexprdata)->persp), SCIPblkmem(scip), nchildren);

   for( c = 0; c < nchildren; ++c )
   {
      /* check if variables are semicontinuous */
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &varexprs, SCIPgetNTotalVars(scip)) );
      SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, children[c], varexprs, &nvars) );
      bvar = NULL;
      expr_is_onoff = TRUE;
      if( nvars == 0 )
         expr_is_onoff = FALSE;

      /* all variables should be semicontinuous */
      for( v = 0; v < nvars; ++v )
      {
         var = SCIPgetConsExprExprVarVar(varexprs[v]);
         if( !varIsSemicontinuous(scip, var, &bvar, (*nlhdlrexprdata)->scvars) )
         {
            expr_is_onoff = FALSE;
            break;
         }
      }
      if( !expr_is_onoff )
         SCIP_CALL( nlhdlrexprdataAddLinTerm(scip, *nlhdlrexprdata, coefs[c], children[c]) );
      else
         nlhdlrexprdataAddPerspTerm(scip, conshdlr, *nlhdlrexprdata, coefs[c], children[c], bvar);

      for( v = 0; v < nvars; ++v )
      {
         SCIPreleaseConsExprExpr(scip, &varexprs[v]);
      }
      SCIPfreeBlockMemoryArray(scip, &varexprs, SCIPgetNTotalVars(scip));
   }

   if( strcmp("sum", SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr))) != 0 )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &children, 1);
      SCIPfreeBlockMemoryArrayNull(scip, &coefs, 1);
   }

   if( !SCIPhashmapIsEmpty((*nlhdlrexprdata)->persp) )
   {
      SCIPdebugMsg(scip, "\nFound persp expressions: ");
      for( c = 0; c < SCIPhashmapGetNEntries((*nlhdlrexprdata)->persp); ++c )
      {
         SCIP_HASHMAPENTRY* entry = SCIPhashmapGetEntry((*nlhdlrexprdata)->persp, c);
         if( entry != NULL )
         {
            pexpr = (SCIP_CONSEXPR_EXPR*) SCIPhashmapEntryGetImage(entry);
#ifdef SCIP_DEBUG
            SCIPdebugMsg(scip, "\n");
            SCIPprintVar(scip, (SCIP_VAR*)SCIPhashmapEntryGetOrigin(entry), NULL);
            SCIPdebugMsg(scip, ": ");
            SCIPprintConsExprExpr(scip, conshdlr, pexpr, NULL);
            SCIPdebugMsg(scip, "; ");
#endif

            child_curvature = SCIPgetConsExprExprCurvature(pexpr);
            if( (*nlhdlrexprdata)->curvature == SCIP_EXPRCURV_UNKNOWN && (child_curvature == SCIP_EXPRCURV_CONVEX || child_curvature == SCIP_EXPRCURV_CONCAVE) )
               (*nlhdlrexprdata)->curvature = child_curvature;
            else if( (child_curvature != SCIP_EXPRCURV_CONVEX && child_curvature != SCIP_EXPRCURV_CONCAVE) || child_curvature != (*nlhdlrexprdata)->curvature )
            {
               SCIPdebugMsg(scip, "\nNon-convex, curv %d, expr curv %d: removing", child_curvature, (*nlhdlrexprdata)->curvature);
               children = SCIPgetConsExprExprChildren(pexpr);
               coefs = SCIPgetConsExprExprSumCoefs(pexpr);
               nchildren = SCIPgetConsExprExprNChildren(pexpr);
               for( v = 0; v < nchildren; ++v )
               {
                  nlhdlrexprdataAddLinTerm(scip, *nlhdlrexprdata, coefs[v], children[v]);
               }
               SCIPhashmapRemove((*nlhdlrexprdata)->persp, SCIPhashmapEntryGetOrigin(entry));
               SCIPreleaseConsExprExpr(scip, &pexpr);
               (*nlhdlrexprdata)->nperspterms--;
            }
         }
      }
   }

   /* check whether expression is nonlinear, convex or concave, and is not handled by another nonlinear handler */
   if( (*nlhdlrexprdata)->nperspterms > 0 )
   {
      if( (*nlhdlrexprdata)->curvature == SCIP_EXPRCURV_CONVEX )
      {
         *enforcemethods |= SCIP_CONSEXPR_EXPRENFO_SEPABELOW;
         SCIPinfoMessage(scip, NULL, "\ndetected expr to be convex -> can enforce expr <= auxvar\n");
      }
      else if( (*nlhdlrexprdata)->curvature == SCIP_EXPRCURV_CONCAVE )
      {
         *enforcemethods |= SCIP_CONSEXPR_EXPRENFO_SEPAABOVE;
         SCIPinfoMessage(scip, NULL, "\ndetected expr to be concave -> can enforce expr >= auxvar\n");
      }
      *success = TRUE;
      for( c = 0; c < (*nlhdlrexprdata)->nlinterms; ++c )
      {
         createAuxVar(scip, conshdlr, (*nlhdlrexprdata)->linterms[c]);
      }
   }

   if( !*success )
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
   int i;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(auxvalue != NULL);

   if( strcmp("sum", SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr))) == 0 )
      *auxvalue = SCIPgetConsExprExprSumConstant(expr);
   else
      *auxvalue = 0;

   for( i = 0; i < nlhdlrexprdata->nlinterms; ++i ) /* linear exprs */
      *auxvalue += nlhdlrexprdata->lincoefs[i] * SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(nlhdlrexprdata->linterms[i]));

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
#else
#define nlhdlrInitSepaPerspective NULL
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
#else
#define nlhdlrExitSepaPerspective NULL
#endif


/** nonlinear handler separation callback */
static
SCIP_DECL_CONSEXPR_NLHDLRSEPA(nlhdlrSepaPerspective)
{ /*lint --e{715}*/
   SCIP_ROWPREP* rowprep;
   SCIP_ROW* row;
   SCIP_VAR* auxvar;
   int i, v, nvars;
   SCIP_Real fval, fval0;
   SCIP_Real scalar_prod;
   SCIP_CONSEXPR_EXPR* pexpr;
   SCIP_VAR* var;
   SCIP_Bool success;

   *result = SCIP_DIDNOTFIND;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(conshdlr != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(result != NULL);
   assert(ncuts != NULL);

   *ncuts = 0;

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "sepa method of perspective nonlinear handler called for expr: ");
   SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, expr, NULL) );
#endif

   /* if estimating on non-convex side, then do nothing */
   if( ( overestimate && nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONVEX) ||
       (!overestimate && nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONCAVE) )
   {
      SCIPdebugMsg(scip, "\nEstimating on non-convex side, do nothing");
      return SCIP_OKAY;
   }

   /*
    * add the cut: auxvar >= (x - x0) \nabla f(sol) + (f(sol) - f(x0) - (sol - x0) \nabla f(sol)) z + f(x0),
    * where x is semicontinuous, z is binary and x0 is the value of x when z = 0
    */

   SCIP_CALL( SCIPcreateRowprep(scip, &rowprep, overestimate ? SCIP_SIDETYPE_LEFT : SCIP_SIDETYPE_RIGHT, TRUE) );
   auxvar = SCIPgetConsExprExprAuxVar(expr);
   assert(auxvar != NULL);
   SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, auxvar, -1.0) );

   if( strcmp("sum", SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr))) == 0 )
      SCIPaddRowprepConstant(rowprep, SCIPgetConsExprExprSumConstant(expr));

   /* handle linear terms */
   for( i = 0; i < nlhdlrexprdata->nlinterms; ++i )
   {
      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, SCIPgetConsExprExprAuxVar(nlhdlrexprdata->linterms[i]),
                                    nlhdlrexprdata->lincoefs[i]) );
   }

   for( i = 0; i < SCIPhashmapGetNEntries(nlhdlrexprdata->persp); ++i )
   {
      SCIP_HASHMAPENTRY* entry = SCIPhashmapGetEntry(nlhdlrexprdata->persp, i);
      SCIP_SOL* sol0;
      SCIP_Real* vals0;
      SCIP_CONSEXPR_EXPR** varexprs;
      SCIP_VAR** vars;

      if( entry == NULL )
         continue;
      pexpr = (SCIP_CONSEXPR_EXPR*) SCIPhashmapEntryGetImage(entry);
      SCIP_CALL( SCIPcreateSol(scip, &sol0, NULL) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &varexprs, SCIPgetNTotalVars(scip)) );
      SCIP_CALL( SCIPgetConsExprExprVarExprs(scip, conshdlr, pexpr, varexprs, &nvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &vals0, nvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &vars, nvars) );
#ifdef SCIP_DEBUG
      SCIPdebugMsg(scip, "\nbvar = ");
      SCIPprintVar(scip, SCIPhashmapEntryGetOrigin(entry), NULL);
      SCIPdebugMsg(scip, NULL, "\npexpr = ");
      SCIPprintConsExprExpr(scip, conshdlr, pexpr, NULL);
#endif

      /* get x0 */
      for( v = 0; v < nvars; ++v )
      {
         SCIP_SCVARDATA* vardata;
         vars[v] = SCIPgetConsExprExprVarVar(varexprs[v]);
         vardata = (SCIP_SCVARDATA*)SCIPhashmapGetImage(nlhdlrexprdata->scvars, (void*)vars[v]);
         vals0[v] = vardata->val0;
      }
      SCIPsetSolVals(scip, sol0, nvars, vars, vals0);

      /* get f(x0) */
      SCIPevalConsExprExpr(scip, conshdlr, pexpr, sol0, 0);
      fval0 = SCIPgetConsExprExprValue(pexpr);
      SCIPfreeSol(scip, &sol0);

      /* evaluation error or a too large constant -> skip */
      if( SCIPisInfinity(scip, REALABS(fval0)) )
      {
         SCIPdebugMsg(scip, "evaluation error / too large value (%g) for %p\n", fval0, (void*)pexpr);
         return SCIP_OKAY;
      }

      /* get f(sol) */
      SCIPevalConsExprExpr(scip, conshdlr, pexpr, sol, 0);
      fval = SCIPgetConsExprExprValue(pexpr);

      /* evaluation error or a too large constant -> skip */
      if( SCIPisInfinity(scip, REALABS(fval)) )
      {
         SCIPdebugMsg(scip, "evaluation error / too large value (%g) for %p\n", fval, (void*)pexpr);
         return SCIP_OKAY;
      }

      /* add (f(sol) - f(x0))z + f(x0) */
      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, (SCIP_VAR*)SCIPhashmapEntryGetOrigin(entry), fval - fval0) );
      SCIPaddRowprepConstant(rowprep, fval0);

      /* compute gradient */
      SCIP_CALL( SCIPcomputeConsExprExprGradient(scip, conshdlr, pexpr, sol, 0) );

      /* gradient evaluation error -> skip */
      if( SCIPgetConsExprExprDerivative(pexpr) == SCIP_INVALID ) /*lint !e777*/
      {
         SCIPdebugMsg(scip, "gradient evaluation error for %p\n", (void*)pexpr);
         return SCIP_OKAY;
      }

      scalar_prod = 0.0;
      for(v = 0; v < nvars; ++v)
      {
         var = SCIPgetConsExprExprVarVar(varexprs[v]);

         /* add xi f'xi(sol) */
         SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, var, SCIPgetConsExprExprPartialDiff(scip, conshdlr, pexpr, var)) );
         /* add -x0i f'xi(sol) */
         SCIPaddRowprepConstant(rowprep, -vals0[v]*SCIPgetConsExprExprPartialDiff(scip, conshdlr, pexpr, var));

         /* compute -(soli - x0i) f'xi(sol) */
         scalar_prod -= (SCIPgetSolVal(scip, sol, var) - vals0[v])*SCIPgetConsExprExprPartialDiff(scip, conshdlr, pexpr, var);
         SCIPreleaseConsExprExpr(scip, &varexprs[v]);
      }
      SCIPfreeBlockMemoryArray(scip, &varexprs, SCIPgetNTotalVars(scip));
      SCIPfreeBlockMemoryArray(scip, &vals0, nvars);
      SCIPfreeBlockMemoryArray(scip, &vars, nvars);

      /* add -(sol - x0) \nabla f(sol)) z */
      SCIPaddRowprepTerm(scip, rowprep, (SCIP_VAR*)SCIPhashmapEntryGetOrigin(entry), scalar_prod);
   }

   /* merge coefficients that belong to same variable */
   SCIPmergeRowprepTerms(scip, rowprep);

   rowprep->local = FALSE;
   SCIP_CALL( SCIPcleanupRowprep(scip, rowprep, sol, SCIP_CONSEXPR_CUTMAXRANGE, mincutviolation, NULL, &success) );

   /* if cut looks good (numerics ok and cutting off solution), then turn into row and add to sepastore */
   if( success )
   {
      SCIP_Bool infeasible;

      SCIP_CALL( SCIPgetRowprepRowCons(scip, &row, rowprep, conshdlr) );

#ifdef SCIP_DEBUG
      SCIPdebugMsg(scip, "\nAt sol point ");
      SCIPprintSol(scip, sol, NULL, TRUE);
#endif
      SCIPinfoMessage(scip, NULL, "\nadding cut ");
      SCIP_CALL( SCIPprintRow(scip, row, NULL) );

      SCIP_CALL( SCIPaddRow(scip, row, FALSE, &infeasible) );
      *ncuts++;

      if( infeasible )
      {
         *result = SCIP_CUTOFF;
         ++nlhdlr->ncutoffs;
      }
      else
      {
         *result = SCIP_SEPARATED;
         ++nlhdlr->ncutsfound;
      }

      SCIP_CALL( SCIPreleaseRow(scip, &row) );
   }

   SCIPfreeRowprep(scip, &rowprep);

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
#else
#define nlhdlrEstimatePerspective NULL
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
#else
#define nlhdlrIntevalPerspective NULL
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
#else
#define nlhdlrReversepropPerspective NULL
#endif


/** nonlinear handler callback for branching scores */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE(nlhdlrBranchscorePerspective)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of perspective nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrBranchscorePerspective NULL
#endif


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
#else
#define nlhdlrReformulatePerspective NULL
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
   SCIPsetConsExprNlhdlrInitExit(scip, nlhdlr, nlhdlrInitPerspective, nlhdlrExitPerspective);
   SCIPsetConsExprNlhdlrSepa(scip, nlhdlr, NULL, nlhdlrSepaPerspective, NULL, NULL);

   return SCIP_OKAY;
}
