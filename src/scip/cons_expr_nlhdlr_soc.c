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

/**@file   cons_expr_nlhdlr_soc.h
 * @brief  nonlinear handler for second order cones
 * @author Benjamin Mueller
 */

#include <string.h>

#include "scip/cons_expr_nlhdlr_soc.h"
#include "scip/cons_expr.h"
#include "scip/cons_expr_pow.h"
#include "scip/cons_expr_sum.h"

/* fundamental nonlinear handler properties */
#define NLHDLR_NAME         "soc"
#define NLHDLR_DESC         "soc nonlinear handler"
#define NLHDLR_PRIORITY     100

/*
 * Data structures
 */

/** nonlinear handler expression data */
struct SCIP_ConsExpr_NlhdlrExprData
{
   SCIP_CONSEXPR_EXPR**  exprs;             /**< expressions that appear in the SQRT */
   SCIP_Real*            coefs;             /**< coefficients of each expression */
   int                   nexprs;            /**< total number of expressions */
   SCIP_Real             constant;          /**< constant term in the SQRT */
   SCIP_VAR*             rhsvar;            /**< right-hand side variable */
   SCIP_Real             rhscoef;           /**< coefficient of right-hand side variable */
};

/** nonlinear handler data */
struct SCIP_ConsExpr_NlhdlrData
{
};

/*
 * Local methods
 */

/** helper method to create nonlinear handler expression data */
static
SCIP_RETCODE createNlhdlrExprData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_CONSEXPR_EXPR**  exprs,              /**< expressions */
   SCIP_Real*            coefs,              /**< coefficients for each expression */
   int                   nexprs,             /**< total number of expressions */
   SCIP_Real             constant,           /**< constant */
   SCIP_VAR*             rhsvar,             /**< right-hand side variable */
   SCIP_Real             rhscoef,            /**< coefficient of right-hand side variable */
   SCIP_CONSEXPR_NLHDLREXPRDATA** nlhdlrexprdata /**< pointer to store nonlinear handler expression data */
   )
{
   int i;

   assert(conshdlr != NULL);
   assert(expr != NULL);
   assert(exprs != NULL);
   assert(coefs != NULL);
   assert(rhsvar != NULL);
   assert(rhscoef != 0.0);
   assert(nexprs > 1);
   assert(nlhdlrexprdata != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, nlhdlrexprdata) );
   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(*nlhdlrexprdata)->exprs, exprs, nexprs) );
   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(*nlhdlrexprdata)->coefs, coefs, nexprs) );
   (*nlhdlrexprdata)->nexprs = nexprs;
   (*nlhdlrexprdata)->constant = constant;
   (*nlhdlrexprdata)->rhsvar = rhsvar;
   (*nlhdlrexprdata)->rhscoef = rhscoef;

   /* capture expressions */
   for( i = 0; i < nexprs; ++i )
   {
      assert(exprs[i] != NULL);
      SCIPcaptureConsExprExpr(exprs[i]);
   }

   /* capture RHS variable */
   SCIP_CALL( SCIPcaptureVar(scip, rhsvar) );

   return SCIP_OKAY;
}

/** helper method to free nonlinear handler expression data */
static
SCIP_RETCODE freeNlhdlrExprData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLREXPRDATA** nlhdlrexprdata /**< pointer to free nonlinear handler expression data */
   )
{
   int i;

   assert(nlhdlrexprdata != NULL);
   assert(*nlhdlrexprdata != NULL);
   assert((*nlhdlrexprdata)->nexprs > 1);

   /* release RHS variable */
   SCIP_CALL( SCIPreleaseVar(scip, &(*nlhdlrexprdata)->rhsvar) );

   /* release expressions */
   for( i = 0; i < (*nlhdlrexprdata)->nexprs; ++i )
   {
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &(*nlhdlrexprdata)->exprs[i]) );
   }

   SCIPfreeBlockMemoryArray(scip, &(*nlhdlrexprdata)->coefs, (*nlhdlrexprdata)->nexprs);
   SCIPfreeBlockMemoryArray(scip, &(*nlhdlrexprdata)->exprs, (*nlhdlrexprdata)->nexprs);
   SCIPfreeBlockMemory(scip, nlhdlrexprdata);

   return SCIP_OKAY;
}

/** helper method to detect SQRT(sum_i (expr_i)^2 + const) <= auxvar */
static
SCIP_RETCODE detectSocNorm(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_VAR*             auxvar,             /**< auxiliary variable */
   SCIP_CONSEXPR_NLHDLREXPRDATA** nlhdlrexprdata, /**< pointer to store nonlinear handler expression data */
   SCIP_Bool*            success             /**< pointer to store whether SOC structure has been detected */
   )
{
   SCIP_CONSEXPR_EXPR** children;
   SCIP_CONSEXPR_EXPR* child;
   int nchildren;
   int i;

   assert(conshdlr != NULL);
   assert(expr != NULL);
   assert(auxvar != NULL);
   assert(success != NULL);

   *success = FALSE;

   /* relation is not "<=" -> skip */
   if( SCIPisInfinity(scip, SCIPvarGetUbLocal(auxvar)) )
      return SCIP_OKAY;

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);

   /* check whether expression is a SQRT and has a sum as children with at least 2 children */
   if( SCIPgetConsExprExprHdlr(expr) != SCIPgetConsExprExprHdlrPow(conshdlr) || SCIPgetConsExprExprPowExponent(expr) != 0.5
      || SCIPgetConsExprExprHdlr(child) != SCIPgetConsExprExprHdlrSum(conshdlr) || SCIPgetConsExprExprNChildren(child) < 2 )
      return SCIP_OKAY;

   /* get children of the sum */
   children = SCIPgetConsExprExprChildren(child);
   nchildren = SCIPgetConsExprExprNChildren(child);

   for( i = 0; i < nchildren; ++i )
   {
      if( SCIPgetConsExprExprHdlr(children[i]) != SCIPgetConsExprExprHdlrPow(conshdlr) || SCIPgetConsExprExprPowExponent(children[i]) != 2.0 )
         return SCIP_OKAY;
   }

   /* found SOC structure -> create required auxiliary variables */
   for( i = 0; i < nchildren; ++i )
   {
      SCIP_CALL( SCIPcreateConsExprExprAuxVar(scip, conshdlr, children[i], NULL) );
   }

   *success = TRUE;

   /* create and store nonlinear handler expression data */
   SCIP_CALL( createNlhdlrExprData(scip, conshdlr, expr, children, SCIPgetConsExprExprSumCoefs(child),
      nchildren, SCIPgetConsExprExprSumConstant(child), auxvar, 1.0, nlhdlrexprdata) );
   assert(*nlhdlrexprdata != NULL);

#ifdef SCIP_DEBUG
   SCIPdebugMsg(scip, "found SOC structure for expression %p\n", (void*)expr);
   SCIPprintConsExprExpr(scip, conshdlr, expr, NULL);
   SCIPinfoMessage(scip, NULL, " <= %s\n", SCIPvarGetName(auxvar));
#endif

   return SCIP_OKAY;
}

/** helper method to detect SOC structures */
static
SCIP_RETCODE detectSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression */
   SCIP_VAR*             auxvar,             /**< auxiliary variable */
   SCIP_CONSEXPR_NLHDLREXPRDATA** nlhdlrexprdata, /**< pointer to store nonlinear handler expression data */
   SCIP_Bool*            success             /**< pointer to store whether SOC structure has been detected */
   )
{
   SCIP_CONSHDLR* conshdlr;

   assert(expr != NULL);
   assert(auxvar != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(success != NULL);

   /* no expression constraint handler -> skip */
   conshdlr = SCIPfindConshdlr(scip, "expr");
   if( conshdlr == NULL )
   {
      *success = FALSE;
      return SCIP_OKAY;
   }

   /* check whether expression is given as */
   SCIP_CALL( detectSocNorm(scip, conshdlr, expr, auxvar, nlhdlrexprdata, success) );

   return SCIP_OKAY;
}

/*
 * Callback methods of nonlinear handler
 */

/** nonlinear handler copy callback */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRCOPYHDLR(nlhdlrCopyhdlrSoc)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrCopyhdlrSoc NULL
#endif

/** callback to free data of handler */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRFREEHDLRDATA(nlhdlrFreehdlrdataSoc)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrFreehdlrdataSoc NULL
#endif


/** callback to free expression specific data */
static
SCIP_DECL_CONSEXPR_NLHDLRFREEEXPRDATA(nlhdlrFreeExprDataSoc)
{  /*lint --e{715}*/
   assert(*nlhdlrexprdata != NULL);

   SCIP_CALL( freeNlhdlrExprData(scip, nlhdlrexprdata) );

   return SCIP_OKAY;
}


/** callback to be called in initialization */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRINIT(nlhdlrInitSoc)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrInitSoc NULL
#endif


/** callback to be called in deinitialization */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLREXIT(nlhdlrExitSoc)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrExitSoc NULL
#endif


/** callback to detect structure in expression tree */
static
SCIP_DECL_CONSEXPR_NLHDLRDETECT(nlhdlrDetectSoc)
{ /*lint --e{715}*/
   SCIP_VAR* auxvar;

   assert(expr != NULL);

   /* TODO is it worth to detect during presolving and then try to apply some bound strengthening? */
   if( SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING )
      return SCIP_OKAY;

   auxvar = SCIPgetConsExprExprAuxVar(expr);
   assert(auxvar != NULL);

   if( isroot )
   {
      SCIP_CALL( detectSOC(scip, expr, auxvar, nlhdlrexprdata, success) );
   }
   else
   {
      *success = FALSE;
   }

   return SCIP_OKAY;
}


/** auxiliary evaluation callback of nonlinear handler */
static
SCIP_DECL_CONSEXPR_NLHDLREVALAUX(nlhdlrEvalauxSoc)
{ /*lint --e{715}*/
   int i;

   assert(nlhdlrexprdata != NULL);
   assert(nlhdlrexprdata->exprs != NULL);
   assert(nlhdlrexprdata->coefs != NULL);
   assert(nlhdlrexprdata->nexprs > 1);

   /* compute sum_i coef_i expr_i^2 + constant */
   *auxvalue = nlhdlrexprdata->constant;

   for( i = 0; i < nlhdlrexprdata->nexprs; ++i )
   {
      SCIP_CONSEXPR_EXPR* child;
      SCIP_VAR* var;

      child = nlhdlrexprdata->exprs[i];
      assert(child != NULL);

      var = SCIPgetConsExprExprAuxVar(child);
      assert(var != NULL);

      *auxvalue += nlhdlrexprdata->coefs[i] * SQR(SCIPgetSolVal(scip, sol, var));
   }
   assert(*auxvalue >= 0.0);

   /* compute ||sum_i coef_i expr_i^2 + constant|| */
   *auxvalue = SQRT(*auxvalue);

   return SCIP_OKAY;
}


/** separation deinitialization method of a nonlinear handler (called during CONSINITLP) */
static
SCIP_DECL_CONSEXPR_NLHDLRINITSEPA(nlhdlrInitSepaSoc)
{ /*lint --e{715}*/
   assert(nlhdlrexprdata != NULL);

   /* create disaggregation variables */

   return SCIP_OKAY;
}


/** separation deinitialization method of a nonlinear handler (called during CONSEXITSOL) */
static
SCIP_DECL_CONSEXPR_NLHDLREXITSEPA(nlhdlrExitSepaSoc)
{ /*lint --e{715}*/
   assert(nlhdlrexprdata != NULL);

   return SCIP_OKAY;
}


/** nonlinear handler separation callback */
static
SCIP_DECL_CONSEXPR_NLHDLRSEPA(nlhdlrSepaSoc)
{ /*lint --e{715}*/

   *result = SCIP_DIDNOTRUN;

   return SCIP_OKAY;
}


/** nonlinear handler under/overestimation callback */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRESTIMATE(nlhdlrEstimateSoc)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrEstimateSoc NULL
#endif


/** nonlinear handler interval evaluation callback */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRINTEVAL(nlhdlrIntevalSoc)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrIntevalSoc NULL
#endif


/** nonlinear handler callback for reverse propagation */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP(nlhdlrReversepropSoc)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrReversepropSoc NULL
#endif


/** nonlinear handler callback for branching scores */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE(nlhdlrBranchscoreSoc)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrBranchscoreSoc NULL
#endif


/** nonlinear handler callback for reformulation */
#if 0
static
SCIP_DECL_CONSEXPR_NLHDLRREFORMULATE(nlhdlrReformulateSoc)
{ /*lint --e{715}*/
   SCIPerrorMessage("method of soc nonlinear handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define nlhdlrReformulateSoc NULL
#endif

/*
 * nonlinear handler specific interface methods
 */

/** includes SOC nonlinear handler to consexpr */
SCIP_RETCODE SCIPincludeConsExprNlhdlrSoc(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_NLHDLRDATA* nlhdlrdata;
   SCIP_CONSEXPR_NLHDLR* nlhdlr;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);

   /* create nonlinear handler data */
   nlhdlrdata = NULL;

   /* TODO: create and store nonlinear handler specific data here */

   SCIP_CALL( SCIPincludeConsExprNlhdlrBasic(scip, consexprhdlr, &nlhdlr, NLHDLR_NAME, NLHDLR_DESC, NLHDLR_PRIORITY, nlhdlrDetectSoc, nlhdlrEvalauxSoc, nlhdlrdata) );
   assert(nlhdlr != NULL);

   SCIPsetConsExprNlhdlrCopyHdlr(scip, nlhdlr, nlhdlrCopyhdlrSoc);
   SCIPsetConsExprNlhdlrFreeHdlrData(scip, nlhdlr, nlhdlrFreehdlrdataSoc);
   SCIPsetConsExprNlhdlrFreeExprData(scip, nlhdlr, nlhdlrFreeExprDataSoc);
   SCIPsetConsExprNlhdlrInitExit(scip, nlhdlr, nlhdlrInitSoc, nlhdlrExitSoc);
   SCIPsetConsExprNlhdlrSepa(scip, nlhdlr, nlhdlrInitSepaSoc, nlhdlrSepaSoc, nlhdlrEstimateSoc, nlhdlrExitSepaSoc);
   SCIPsetConsExprNlhdlrProp(scip, nlhdlr, nlhdlrIntevalSoc, nlhdlrReversepropSoc);
   SCIPsetConsExprNlhdlrBranchscore(scip, nlhdlr, nlhdlrBranchscoreSoc);
   SCIPsetConsExprNlhdlrReformulate(scip, nlhdlr, nlhdlrReformulateSoc);

   return SCIP_OKAY;
}
