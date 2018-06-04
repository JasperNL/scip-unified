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

/**@file   cons_expr_nlhdlr_default.c
 * @brief  default nonlinear handler that calls expression handler methods
 * @author Stefan Vigerske
 *
 */

#include <string.h>

#include "scip/cons_expr_nlhdlr_default.h"
#include "scip/cons_expr.h"

/* fundamental nonlinear handler properties */
#define NLHDLR_NAME         "default"
#define NLHDLR_DESC         "default handler for expressions"
#define NLHDLR_PRIORITY     0

/** evaluates an expression w.r.t. the values in the auxiliary variables */
static
SCIP_RETCODE evalExprInAux(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression to be evaluated */
   SCIP_Real*            val,                /**< buffer to store value of expression */
   SCIP_SOL*             sol                 /**< solution to be evaluated */
)
{
   SCIP_Real* childvals;
   SCIP_VAR* childvar;
   int c;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(val != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) > 0);

   SCIP_CALL( SCIPallocBufferArray(scip, &childvals, SCIPgetConsExprExprNChildren(expr)) );

   for( c = 0; c < SCIPgetConsExprExprNChildren(expr); ++c )
   {
      childvar = SCIPgetConsExprExprAuxVar(SCIPgetConsExprExprChildren(expr)[c]);
      assert(childvar != NULL); /* because we created auxvars in detect for every child */

      childvals[c] = SCIPgetSolVal(scip, sol, childvar);
   }

   SCIP_CALL( SCIPevalConsExprExprHdlr(scip, expr, val, childvals, sol) );

   SCIPfreeBufferArray(scip, &childvals);

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRDETECT(nlhdlrDetectDefault)
{ /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRENFO_METHOD mymethods;
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;
   int c;

   assert(scip != NULL);
   assert(nlhdlr != NULL);
   assert(expr != NULL);
   assert(enforcemethods != NULL);
   assert(enforcedbelow != NULL);
   assert(enforcedabove != NULL);
   assert(success != NULL);
   assert(nlhdlrexprdata != NULL);

   *success = FALSE;
   mymethods = SCIP_CONSEXPR_EXPRENFO_NONE;

   exprhdlr = SCIPgetConsExprExprHdlr(expr);
   assert(exprhdlr != NULL);

   /* return interval evaluation possibility if exprhdlr for expr has a inteval callback and no one already provides (a good) inteval */
   if( SCIPhasConsExprExprHdlrIntEval(exprhdlr) && (*enforcemethods & SCIP_CONSEXPR_EXPRENFO_INTEVAL) == 0 )
   {
      mymethods |= SCIP_CONSEXPR_EXPRENFO_INTEVAL;
      *success = TRUE;
   }

   /* return reverse propagation possibility if exprhdlr for expr has a reverseprop callback and no one already provides (a good) reverseprop */
   if( SCIPhasConsExprExprHdlrReverseProp(exprhdlr) && (*enforcemethods & SCIP_CONSEXPR_EXPRENFO_REVERSEPROP) == 0 )
   {
      /* one could claim that reverse propagation is sufficient for enforcement, but separation is probably stronger
       * so, not setting enforcedbelow/above to TRUE here for now
       */
      mymethods |= SCIP_CONSEXPR_EXPRENFO_REVERSEPROP;
      *success = TRUE;
   }

   /* return sepa possibility if exprhdlr for expr has a sepa callback and enforcement is not ensured already */
   if( SCIPhasConsExprExprHdlrSepa(exprhdlr) && (!*enforcedbelow || !*enforcedabove) )
   {
      /* make sure that an (auxiliary) variable exists for every child */
      for( c = 0; c < SCIPgetConsExprExprNChildren(expr); ++c )
      {
         /* todo skip this for value-expressions? would then need update in evalExprInAux, too */
         SCIP_CALL( SCIPcreateConsExprExprAuxVar(scip, conshdlr, SCIPgetConsExprExprChildren(expr)[c], NULL) );
      }

      /* communicate back what the nlhdlr will do
       * - it will enforce via separation on those sides that are not enforced yet
       * - it will provide branching scores if it does separation
       * - it needs to be called for this expression (success = TRUE)
       */
      if( !*enforcedbelow )
      {
         mymethods |= SCIP_CONSEXPR_EXPRENFO_SEPABELOW | SCIP_CONSEXPR_EXPRENFO_BRANCHSCORE;
         *enforcedbelow = TRUE;
         *success = TRUE;
      }

      if( !*enforcedabove )
      {
         mymethods |= SCIP_CONSEXPR_EXPRENFO_SEPAABOVE | SCIP_CONSEXPR_EXPRENFO_BRANCHSCORE;
         *enforcedabove = TRUE;
         *success = TRUE;
      }
   }
#if 0 /* TODO branching method needs to distinguish whether we do separation (thus added auxvar) or only propagate (no auxvar) */
   else if( (!*enforcedbelow || !*enforcedabove) &&
      (mymethods & SCIP_CONSEXPR_EXPRENFO_INTEVAL) != 0 &&
      (mymethods & SCIP_CONSEXPR_EXPRENFO_REVERSEPROP) != 0 )
   {
      /* return branching score possibility if enforcement is not ensured yet, but we provide propagation,
       * since propagation and branching should be sufficient for enforcement, too
       */
      mymethods |= SCIP_CONSEXPR_EXPRENFO_BRANCHSCORE;
      *enforcedbelow = TRUE;
      *enforcedabove = TRUE;
      *success = TRUE;
   }
#endif

   /* it does not makes much sense to advertise a brscore callback if we do not also enforce via separation or propagation */

   if( *success )
   {
      /* remember in the nlhdlr exprdata (pointer) which methods we advertised */
      *nlhdlrexprdata = (SCIP_CONSEXPR_NLHDLREXPRDATA*)(size_t)mymethods;
      /* augment mymethods in enforcemethods */
      *enforcemethods |= mymethods;
   }

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLREVALAUX(nlhdlrEvalAuxDefault)
{ /*lint --e{715}*/
   assert(expr != NULL);
   assert(auxvalue != NULL);

   if( ((SCIP_CONSEXPR_EXPRENFO_METHOD)(size_t)nlhdlrexprdata & SCIP_CONSEXPR_EXPRENFO_SEPABOTH) == 0 )
   {
      /* if we did not say that we separated, then we did not introduce auxvars
       * in that case, return the expression value, though it is a bit odd that we are still called
       */
      *auxvalue = SCIPgetConsExprExprValue(expr);

      return SCIP_OKAY;
   }

   SCIP_CALL( evalExprInAux(scip, expr, auxvalue, sol) );

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRINITSEPA(nlhdlrInitSepaDefault)
{ /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);

   /* if we will not separate, then don't call initsepa */
   if( ((SCIP_CONSEXPR_EXPRENFO_METHOD)(size_t)nlhdlrexprdata & SCIP_CONSEXPR_EXPRENFO_SEPABOTH) == 0 )
      return SCIP_OKAY;

   /* call the separation initialization callback of the expression handler */
   SCIP_CALL( SCIPinitsepaConsExprExprHdlr(scip, conshdlr, expr, overestimate, underestimate, infeasible) );

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRSEPA(nlhdlrSepaDefault)
{ /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);
   assert(result != NULL);
   assert(ncuts != NULL);

   /* if we did not say that we will separate, then stand by it */
   if( ((SCIP_CONSEXPR_EXPRENFO_METHOD)(size_t)nlhdlrexprdata & SCIP_CONSEXPR_EXPRENFO_SEPABOTH) == 0 )
      return SCIP_OKAY;

   if( separated )
   {
      /* don't do anything if someone already separated */
      *result = SCIP_DIDNOTFIND;
      *ncuts = 0;

      return SCIP_OKAY;
   }

   /* call the separation callback of the expression handler */
   SCIP_CALL( SCIPsepaConsExprExprHdlr(scip, conshdlr, expr, sol, overestimate, mincutviolation, result, ncuts) );

   return SCIP_OKAY;
}


static
SCIP_DECL_CONSEXPR_NLHDLREXITSEPA(nlhdlrExitSepaDefault)
{ /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);

   /* if we have not separated, then don't call exitsepa */
   if( ((SCIP_CONSEXPR_EXPRENFO_METHOD)(size_t)nlhdlrexprdata & SCIP_CONSEXPR_EXPRENFO_SEPABOTH) == 0 )
      return SCIP_OKAY;

   /* call the separation deinitialization callback of the expression handler */
   SCIP_CALL( SCIPexitsepaConsExprExprHdlr(scip, expr) );

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRINTEVAL(nlhdlrIntevalDefault)
{ /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);

   /* call the interval evaluation callback of the expression handler */
   SCIP_CALL( SCIPintevalConsExprExprHdlr(scip, expr, interval, varboundrelax) );

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP(nlhdlrReversepropDefault)
{ /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);

   /* call the reverse propagation callback of the expression handler */
   SCIP_CALL( SCIPreversepropConsExprExprHdlr(scip, expr, reversepropqueue, infeasible, nreductions, force) );

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE(nlhdlrBranchscoreDefault)
{ /*lint --e{715}*/
   SCIP_Real auxval; /* value of expression in aux. variables */
   SCIP_Real violation;
   SCIP_CONSEXPR_EXPRENFO_METHOD enfomethods;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(success != NULL);

   enfomethods = (SCIP_CONSEXPR_EXPRENFO_METHOD)(size_t)nlhdlrexprdata;

   /* if we did not say that we will provide branching scores, then stand by it */
   if( (enfomethods & SCIP_CONSEXPR_EXPRENFO_BRANCHSCORE) == 0 )
      return SCIP_OKAY;

   /* call the branching callback of the expression handler */
   SCIP_CALL( SCIPbranchscoreConsExprExprHdlr(scip, expr, sol, auxvalue, brscoretag, success) );

   if( *success )
      return SCIP_OKAY;

   /* fallback: register violation w.r.t. values in auxiliary variables as branching score for each child */

   if( auxvalue == SCIP_INVALID ) /*lint !e777*/
   {
      /* if cannot evaluate, then always branch */
      violation = SCIPinfinity(scip);
   }
   else
   {
      /* get value of auxiliary variable of this expression */
      assert(SCIPgetConsExprExprAuxVar(expr) != NULL);
      auxval = SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(expr));

      /* compute the violation
       * if we said we separate below, then we enforce expr <= auxval, so violation is (positive part of) auxvalue - auxval
       * if we said we separate above, then we enforce expr >= auxval, so violation is (positive part of) auxval - auxvalue
       */
      violation = 0.0;
      if( enfomethods & SCIP_CONSEXPR_EXPRENFO_SEPABELOW )
         violation = MAX(0.0, auxvalue - auxval);
      if( enfomethods & SCIP_CONSEXPR_EXPRENFO_SEPAABOVE )
         violation = MAX(violation, auxval - auxvalue);
   }

   /* if there is a violation, then register it as branching score for each child */
   if( violation > 0.0 )
   {
      int c;

      /* add violation as branching score to all children */
      for( c = 0; c < SCIPgetConsExprExprNChildren(expr); ++c )
         SCIPaddConsExprExprBranchScore(scip, SCIPgetConsExprExprChildren(expr)[c], brscoretag, REALABS(violation));

      *success = TRUE;

      /* count this branchscore as belonging to the exprhdlr, too
       * thus, it will be counted for the default nlhdlr, but also for this exprhdlr
       */
      SCIPincrementConsExprExprHdlrNBranchScore(SCIPgetConsExprExprHdlr(expr));
   }

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRCOPYHDLR(nlhdlrCopyhdlrDefault)
{ /*lint --e{715}*/
   assert(targetscip != NULL);
   assert(targetconsexprhdlr != NULL);
   assert(sourcenlhdlr != NULL);
   assert(strcmp(SCIPgetConsExprNlhdlrName(sourcenlhdlr), NLHDLR_NAME) == 0);

   SCIP_CALL( SCIPincludeConsExprNlhdlrDefault(targetscip, targetconsexprhdlr) );

   return SCIP_OKAY;
}

/** includes default nonlinear handler to consexpr */
SCIP_RETCODE SCIPincludeConsExprNlhdlrDefault(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_NLHDLR* nlhdlr;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);

   SCIP_CALL( SCIPincludeConsExprNlhdlrBasic(scip, consexprhdlr, &nlhdlr, NLHDLR_NAME, NLHDLR_DESC, NLHDLR_PRIORITY, nlhdlrDetectDefault, nlhdlrEvalAuxDefault, NULL) );
   assert(nlhdlr != NULL);

   SCIPsetConsExprNlhdlrCopyHdlr(scip, nlhdlr, nlhdlrCopyhdlrDefault);
   SCIPsetConsExprNlhdlrSepa(scip, nlhdlr, nlhdlrInitSepaDefault, nlhdlrSepaDefault, nlhdlrExitSepaDefault);
   SCIPsetConsExprNlhdlrProp(scip, nlhdlr, nlhdlrIntevalDefault, nlhdlrReversepropDefault);
   SCIPsetConsExprNlhdlrBranchscore(scip, nlhdlr, nlhdlrBranchscoreDefault);

   return SCIP_OKAY;
}
