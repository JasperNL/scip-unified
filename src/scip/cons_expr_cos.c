/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2016 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  CIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_expr_cos.c
 * @brief  handler for cosine expressions
 * @author Fabian Wegscheider
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#define _USE_MATH_DEFINES   /* to get M_PI on Windows */  /*lint !750 */

#include <string.h>
#include <math.h>
#include "scip/cons_expr_cos.h"
#include "scip/cons_expr_sin.h"
#include "cons_expr_value.h"

/* fundamental expression handler properties */
#define EXPRHDLR_NAME         "cos"
#define EXPRHDLR_DESC         "cosine expression"
#define EXPRHDLR_PRECEDENCE   9200
#define EXPRHDLR_HASHKEY      SCIPcalcFibHash(82463.0)

/*
 * Local methods
 */

/** helper function to create cuts for point- or initial separation
 *
 *  A total of 6 different cuts can be generated. All except soltangent are independent of a specific solution and
 *  use only the bounds of the child variable. If their pointers are passed with NULL, the respective coputation
 *  is not performed at all. If one of the computations failes or turns out to be irrelevant, the respective argument
 *  pointer is set to NULL.
 */
static
SCIP_RETCODE computeCutsCos(
        SCIP*                 scip,               /**< SCIP data structure */
        SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
        SCIP_CONSEXPR_EXPR*   expr,               /**< sum expression */
        SCIP_SOL*             sol,                /**< solution to be separated (NULL for the LP solution) */
        SCIP_ROW**            secant,             /**< pointer to store the secant */
        SCIP_ROW**            ltangent,           /**< pointer to store the left tangent */
        SCIP_ROW**            rtangent,           /**< pointer to store the right tangent */
        SCIP_ROW**            lmidtangent,        /**< pointer to store the left middle tangent */
        SCIP_ROW**            rmidtangent,        /**< pointer to store the right middle tangent */
        SCIP_ROW**            soltangent,         /**< pointer to store the solution tangent */
        SCIP_Bool             underestimate       /**< whether the cuts should be underestimating */
)
{
   SCIP_CONSEXPR_EXPR* child;
   SCIP_VAR* auxvar;
   SCIP_VAR* childvar;
   SCIP_Real lincoef;
   SCIP_Real linconst;
   SCIP_Real childlb;
   SCIP_Real childub;
   SCIP_Real lhs;
   SCIP_Real rhs;
   SCIP_Bool success;
   char name[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), "expr") == 0);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), EXPRHDLR_NAME) == 0);

   /* get expression data */
   auxvar = SCIPgetConsExprExprLinearizationVar(expr);
   assert(auxvar != NULL);
   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);
   childvar = SCIPgetConsExprExprLinearizationVar(child);
   assert(childvar != NULL);

   /* shift bounds so that sine functions can be used for computation */
   childlb = SCIPvarGetLbLocal(childvar) + 0.5*M_PI;
   childub = SCIPvarGetUbLocal(childvar) + 0.5*M_PI;
   assert(SCIPisLE(scip, childlb, childub));

   /* if variable is fixed, it does not make sense to add cuts */
   if( SCIPisEQ(scip, childlb, childub) )
      return SCIP_OKAY;

   /*
    * Compute all cuts that where specified upon call.
    * For each linear equation z = a*x + b with bounds [lb,ub] the parameters can be computed by:
    *
    * a = -sin(x^)    and     b = cos(x^) - a * x^        where x^ is any known point in [lb,ub]
    *
    * and the resulting cut is       a*x - z <=/>= -b           depending on over-/underestimation
    */

   /* compute secant between lower and upper bound */
   if( secant != NULL )
   {
      *secant = NULL;

      if( underestimate )
         success = SCIPcomputeSecantSin(scip, &lincoef, &linconst, childlb, childub);
      else
         success = SCIPcomputeSecantSin(scip, &lincoef, &linconst, -childub, -childlb);

      if( success )
      {
         /* shift cut back */
         linconst += lincoef  * 0.5 * M_PI;

         lhs = underestimate ? -SCIPinfinity(scip) : linconst;
         rhs = underestimate ? -linconst : SCIPinfinity(scip);

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "cos_secant_%s", SCIPvarGetName(childvar));

         SCIP_CALL( SCIPcreateEmptyRowCons(scip, secant, conshdlr, name, lhs, rhs,
                                           TRUE, FALSE, FALSE) );

         SCIP_CALL( SCIPaddVarToRow(scip, *secant, auxvar, -1.0) );
         SCIP_CALL( SCIPaddVarToRow(scip, *secant, childvar, lincoef) );
      }
   }

   /* compute tangent at lower bound */
   if( ltangent != NULL )
   {
      *ltangent = NULL;

      if( underestimate )
         success = SCIPcomputeLeftTangentSin(scip, &lincoef, &linconst, childlb);
      else
         success = SCIPcomputeRightTangentSin(scip, &lincoef, &linconst, -childlb);

      if( success )
      {
         /* shift cut back */
         linconst += lincoef  * 0.5 * M_PI;

         lhs = underestimate ? -SCIPinfinity(scip) : linconst;
         rhs = underestimate ? -linconst : SCIPinfinity(scip);

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "cos_ltangent_%s", SCIPvarGetName(childvar));

         SCIP_CALL( SCIPcreateEmptyRowCons(scip, ltangent, conshdlr, name, lhs, rhs,
                                           TRUE, FALSE, FALSE) );

         SCIP_CALL( SCIPaddVarToRow(scip, *ltangent, auxvar, -1.0) );
         SCIP_CALL( SCIPaddVarToRow(scip, *ltangent, childvar, lincoef) );
      }
   }

   /* compute tangent at upper bound */
   if( rtangent != NULL )
   {
      *rtangent = NULL;

      if( underestimate )
         success = SCIPcomputeRightTangentSin(scip, &lincoef, &linconst, childub);
      else
         success = SCIPcomputeLeftTangentSin(scip, &lincoef, &linconst, -childub);

      if( success )
      {
         /* shift cut back */
         linconst += lincoef  * 0.5 * M_PI;

         lhs = underestimate ? -SCIPinfinity(scip) : linconst;
         rhs = underestimate ? -linconst : SCIPinfinity(scip);

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "cos_rtangent_%s", SCIPvarGetName(childvar));

         SCIP_CALL( SCIPcreateEmptyRowCons(scip, rtangent, conshdlr, name, lhs, rhs,
                                           TRUE, FALSE, FALSE) );

         SCIP_CALL( SCIPaddVarToRow(scip, *rtangent, auxvar, -1.0) );
         SCIP_CALL( SCIPaddVarToRow(scip, *rtangent, childvar, lincoef) );
      }
   }

   /* compute tangent at solution point */
   if( soltangent != NULL )
   {
      SCIP_Real refpoint;

      *soltangent = NULL;
      refpoint = SCIPgetSolVal(scip, sol, childvar);

      if( underestimate )
         success = SCIPcomputeSolTangentSin(scip, &lincoef, &linconst, childlb, childub, refpoint);
      else
         success = SCIPcomputeSolTangentSin(scip, &lincoef, &linconst, -childub, -childlb, -refpoint);

      if( success )
      {

         /* shift cut back */
         linconst += lincoef  * 0.5 * M_PI;

         lhs = underestimate ? -SCIPinfinity(scip) : linconst;
         rhs = underestimate ? -linconst : SCIPinfinity(scip);

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "cos_soltangent_%s", SCIPvarGetName(childvar));

         SCIP_CALL( SCIPcreateEmptyRowCons(scip, soltangent, conshdlr, name, lhs, rhs,
                                           TRUE, FALSE, FALSE) );

         SCIP_CALL( SCIPaddVarToRow(scip, *soltangent, auxvar, -1.0) );
         SCIP_CALL( SCIPaddVarToRow(scip, *soltangent, childvar, lincoef) );
      }
   }


   /* compute left middle tangent, that is tangent at some other point which goes through (lb,sin(lb))
    * if secant or soltangent are feasible, this cut can never beat them
    */
   if( lmidtangent != NULL && (secant == NULL || *secant == NULL) && (soltangent == NULL || *soltangent == NULL) )
   {
      SCIP_ROW** cutbuffer;
      SCIP_Bool issecant;

      *lmidtangent = NULL;

      if( underestimate )
         success = SCIPcomputeLeftMidTangentSin(scip, &lincoef, &linconst, &issecant, childlb, childub);
      else
         success = SCIPcomputeRightMidTangentSin(scip, &lincoef, &linconst, &issecant, -childub, -childlb);

      /* if the cut connects bounds it is stored in secant */
      cutbuffer = (issecant && secant != NULL) ? secant : lmidtangent;

      if( success )
      {
         /* shift cut back */
         linconst += lincoef  * 0.5 * M_PI;

         lhs = underestimate ? -SCIPinfinity(scip) : linconst;
         rhs = underestimate ? -linconst : SCIPinfinity(scip);

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "cos_lmidtangent_%s", SCIPvarGetName(childvar));

         SCIP_CALL( SCIPcreateEmptyRowCons(scip, cutbuffer, conshdlr, name, lhs, rhs,
                                           TRUE, FALSE, FALSE) );

         SCIP_CALL( SCIPaddVarToRow(scip, *cutbuffer, auxvar, -1.0) );
         SCIP_CALL( SCIPaddVarToRow(scip, *cutbuffer, childvar, lincoef) );
      }
   }
   else if( lmidtangent != NULL )
      *lmidtangent = NULL;

   /* compute right middle tangent, that is tangent at some other point which goes through (ub,sin(ub))
    * if secant or soltangent are feasible, this cut can never beat them
    */
   if( rmidtangent != NULL && (secant == NULL || *secant == NULL) && (soltangent == NULL || *soltangent == NULL) )
   {
      SCIP_ROW** cutbuffer;
      SCIP_Bool issecant;

      *rmidtangent = NULL;

      if( underestimate )
         success = SCIPcomputeRightMidTangentSin(scip, &lincoef, &linconst, &issecant, childlb, childub);
      else
         success = SCIPcomputeLeftMidTangentSin(scip, &lincoef, &linconst, &issecant, -childub, -childlb);

      /* if the cut connects bounds it is stored in secant */
      cutbuffer = (issecant && secant != NULL) ? secant : rmidtangent;

      if( success )
      {

         /* shift cut back */
         linconst += lincoef  * 0.5 * M_PI;

         lhs = underestimate ? -SCIPinfinity(scip) : linconst;
         rhs = underestimate ? -linconst : SCIPinfinity(scip);

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "cos_lmidtangent_%s", SCIPvarGetName(childvar));

         SCIP_CALL( SCIPcreateEmptyRowCons(scip, cutbuffer, conshdlr, name, lhs, rhs,
                                           TRUE, FALSE, FALSE) );

         SCIP_CALL( SCIPaddVarToRow(scip, *cutbuffer, auxvar, -1.0) );
         SCIP_CALL( SCIPaddVarToRow(scip, *cutbuffer, childvar, lincoef) );
      }
   }
   else if( rmidtangent != NULL )
      *rmidtangent = NULL;

   return SCIP_OKAY;
}

/*
 * Callback methods of expression handler
 */

/** expression handler copy callback */
static
SCIP_DECL_CONSEXPR_EXPRCOPYHDLR(copyhdlrCos)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPincludeConsExprExprHdlrCos(scip, consexprhdlr) );
   *valid = TRUE;

   return SCIP_OKAY;
}

/** simplifies a cos expression
 * Evaluates the sine value function when its child is a value expression
 * TODO: add further simplifications
 */
static
SCIP_DECL_CONSEXPR_EXPRSIMPLIFY(simplifyCos)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_CONSHDLR* conshdlr;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(simplifiedexpr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   conshdlr = SCIPfindConshdlr(scip, "expr");
   assert(conshdlr != NULL);

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);

   /* check for value expression */
   if( SCIPgetConsExprExprHdlr(child) == SCIPgetConsExprExprHdlrValue(conshdlr) )
   {
      SCIP_CALL( SCIPcreateConsExprExprValue(scip, conshdlr, simplifiedexpr,
            COS(SCIPgetConsExprExprValueValue(child))) );
   }
   else
   {
      *simplifiedexpr = expr;

      /* we have to capture it, since it must simulate a "normal" simplified call in which a new expression is created */
      SCIPcaptureConsExprExpr(*simplifiedexpr);
   }

   return SCIP_OKAY;
}

/** expression data copy callback */
static
SCIP_DECL_CONSEXPR_EXPRCOPYDATA(copydataCos)
{  /*lint --e{715}*/
   assert(targetscip != NULL);
   assert(targetexprdata != NULL);
   assert(targetexprdata != NULL);
   assert(sourceexpr != NULL);
   assert(SCIPgetConsExprExprData(sourceexpr) == NULL);

   *targetexprdata = NULL;

   return SCIP_OKAY;
}

/** expression data free callback */
static
SCIP_DECL_CONSEXPR_EXPRFREEDATA(freedataCos)
{  /*lint --e{715}*/
   assert(expr != NULL);

   SCIPsetConsExprExprData(expr, NULL);

   return SCIP_OKAY;
}

/** expression print callback */
static
SCIP_DECL_CONSEXPR_EXPRPRINT(printCos)
{  /*lint --e{715}*/
   assert(expr != NULL);

   switch( stage )
   {
   case SCIP_CONSEXPREXPRWALK_ENTEREXPR :
   {
      /* print function with opening parenthesis */
      SCIPinfoMessage(scip, file, "%s(", EXPRHDLR_NAME);
      break;
   }

   case SCIP_CONSEXPREXPRWALK_VISITINGCHILD :
   {
      assert(SCIPgetConsExprExprWalkCurrentChild(expr) == 0);
      break;
   }

   case SCIP_CONSEXPREXPRWALK_LEAVEEXPR :
   {
      /* print closing parenthesis */
      SCIPinfoMessage(scip, file, ")");
      break;
   }

   case SCIP_CONSEXPREXPRWALK_VISITEDCHILD :
   default: ;
   }

   return SCIP_OKAY;
}

/** expression parse callback */
static
SCIP_DECL_CONSEXPR_EXPRPARSE(parseCos)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* childexpr;

   assert(expr != NULL);

   /* parse child expression from remaining string */
   SCIP_CALL( SCIPparseConsExprExpr(scip, consexprhdlr, string, endstring, &childexpr) );
   assert(childexpr != NULL);

   /* create cosine expression */
   SCIP_CALL( SCIPcreateConsExprExprCos(scip, consexprhdlr, expr, childexpr) );
   assert(*expr != NULL);

   /* release child expression since it has been captured by the cosine expression */
   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &childexpr) );

   *success = TRUE;

   return SCIP_OKAY;
}

/** expression (point-) evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPREVAL(evalCos)
{  /*lint --e{715}*/
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(SCIPgetConsExprExprValue(SCIPgetConsExprExprChildren(expr)[0]) != SCIP_INVALID); /*lint !e777*/

   *val = COS(SCIPgetConsExprExprValue(SCIPgetConsExprExprChildren(expr)[0]));

   return SCIP_OKAY;
}

/** expression derivative evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRBWDIFF(bwdiffCos)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;

   assert(expr != NULL);
   assert(idx >= 0 && idx < SCIPgetConsExprExprNChildren(expr));
   assert(SCIPgetConsExprExprValue(expr) != SCIP_INVALID); /*lint !e777*/

   child = SCIPgetConsExprExprChildren(expr)[idx];
   assert(child != NULL);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(child)), "val") != 0);

   *val = -SIN(SCIPgetConsExprExprValue(child));

   return SCIP_OKAY;
}

/** expression interval evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRINTEVAL(intevalCos)
{  /*lint --e{715}*/
   SCIP_INTERVAL childinterval;

   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   childinterval = SCIPgetConsExprExprInterval(SCIPgetConsExprExprChildren(expr)[0]);
   assert(!SCIPintervalIsEmpty(SCIPinfinity(scip), childinterval));

   SCIPintervalCos(SCIPinfinity(scip), interval, childinterval);

   return SCIP_OKAY;
}

/** separation initialization callback */
static
SCIP_DECL_CONSEXPR_EXPRINITSEPA(initSepaCos)
{  /*lint --e{715}*/
   SCIP_ROW* cuts[5];   /* 0: secant, 1: left tangent, 2: right tangent, 3: left mid tangent, 4: right mid tangent */
   int i;

   *infeasible = FALSE;

   /* compute underestimating cuts */
   SCIP_CALL( computeCutsCos(scip, conshdlr, expr, NULL, &cuts[0], &cuts[1], &cuts[2], &cuts[3], &cuts[4], NULL, TRUE) );

   for( i = 0; i < 5; ++i)
   {
      /* only the cuts which could be created are added */
      if( !*infeasible && cuts[i] != NULL )
      {
         SCIP_CALL( SCIPmassageConsExprExprCut(scip, &cuts[i], NULL, -SCIPinfinity(scip)) );

         if( cuts[i] != NULL ) {
            SCIP_CALL( SCIPaddCut(scip, NULL, cuts[i], FALSE, infeasible) );
         }
      }

      /* release the row */
      if( cuts[i] != NULL )
      {
         SCIP_CALL( SCIPreleaseRow(scip, &cuts[i]) );
      }
   }

   /* compute overestimating cuts */
   SCIP_CALL( computeCutsCos(scip, conshdlr, expr, NULL, &cuts[0], &cuts[1], &cuts[2], &cuts[3], &cuts[4], NULL, FALSE) );

   for( i = 0; i < 5; ++i) {
      /* only the cuts which could be created are added */
      if (!*infeasible && cuts[i] != NULL) {
         SCIP_CALL(SCIPmassageConsExprExprCut(scip, &cuts[i], NULL, -SCIPinfinity(scip)));

         if (cuts[i] != NULL) {
            SCIP_CALL(SCIPaddCut(scip, NULL, cuts[i], FALSE, infeasible));
         }
      }

      /* release the row */
      if (cuts[i] != NULL) {
         SCIP_CALL(SCIPreleaseRow(scip, &cuts[i]));
      }
   }

   return SCIP_OKAY;
}

/** expression separation callback */
static
SCIP_DECL_CONSEXPR_EXPRSEPA(sepaCos)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_VAR* auxvar;
   SCIP_VAR* childvar;
   SCIP_ROW* cuts[4] = {NULL, NULL, NULL, NULL};
   SCIP_Real violation;
   SCIP_Bool underestimate;
   SCIP_Bool infeasible;
   int i;

   /* get expression data */
   auxvar = SCIPgetConsExprExprLinearizationVar(expr);
   assert(auxvar != NULL);
   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);
   childvar = SCIPgetConsExprExprLinearizationVar(child);
   assert(childvar != NULL);

   infeasible = FALSE;
   *ncuts = 0;
   *result = SCIP_DIDNOTFIND;

   /* compute the violation; this determines whether we need to over- or underestimate */
   violation = exp(SCIPgetSolVal(scip, sol, childvar)) - SCIPgetSolVal(scip, sol, auxvar);

   /* check if there is a violation */
   if( SCIPisEQ(scip, violation, 0.0) )
      return SCIP_OKAY;

   /* determine if we need to under- or overestimate */
   underestimate = SCIPisGT(scip, violation, 0.0);

   /* compute all possible inequalities; the resulting cuts are stored in the cuts array
    *
    *  - cuts[0] = secant
    *  - cuts[1] = secant connecting (lb,cos(lbx)) with left tangent point
    *  - cuts[1] = secant connecting (ub,cos(ubx)) with right tangent point
    *  - cuts[3] = solution tangent (for convex / concave segments that globally under- / overestimate)
    */
   SCIP_CALL( computeCutsCos(scip, conshdlr, expr, sol, &cuts[0], NULL, NULL, &cuts[1], &cuts[2], &cuts[3],
                             underestimate) );

   for( i = 0; i < 4; ++i )
   {
      if( cuts[i] == NULL )
         continue;

      SCIP_CALL( SCIPmassageConsExprExprCut(scip, &cuts[i], sol, minviolation) );

      if( cuts[i] == NULL )
         continue;

      violation = -SCIPgetRowSolFeasibility(scip, cuts[i], sol);
      if( SCIPisGE(scip, violation, minviolation) )
      {
         SCIP_CALL( SCIPaddCut(scip, sol, cuts[i], FALSE, &infeasible) );
         *ncuts += 1;

         if( infeasible )
         {
            *result = SCIP_CUTOFF;
            break;
         }
         else
            *result = SCIP_SEPARATED;
      }

      /* release the secant */
      if( cuts[i] != NULL )
      {
         SCIP_CALL( SCIPreleaseRow(scip, &cuts[i]) );
      }
   }

   return SCIP_OKAY;
}

/** expression reverse propagation callback */
static
SCIP_DECL_CONSEXPR_REVERSEPROP(reversepropCos)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_INTERVAL interval;
   SCIP_INTERVAL childbound;
   SCIP_Real newinf;
   SCIP_Real newsup;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(nreductions != NULL);
   assert(SCIPintervalGetInf(SCIPgetConsExprExprInterval(expr)) >= -1.0);
   assert(SCIPintervalGetSup(SCIPgetConsExprExprInterval(expr)) <= 1.0);

   *nreductions = 0;

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);

   childbound = SCIPgetConsExprExprInterval(child);
   newinf = childbound.inf;
   newsup = childbound.sup;

   interval = SCIPgetConsExprExprInterval(expr);

   if( !SCIPisInfinity(scip, -newinf) )
   {
      /* l(x) and u(x) are lower/upper bound of child, l(s) and u(s) are lower/upper bound of cos expr
       *
       * if cos(l(x)) < l(s), we are looking for k minimal s.t. a + 2k*pi > l(x) where a = acos(l(s))
       * then the new lower bound is a + 2k*pi
       */
      if( SCIPisLT(scip, COS(newinf), interval.inf) )
      {
         SCIP_Real a = ACOS(interval.inf);
         int k = (int) ceil((newinf - a) / (2.0*M_PI));
         newinf = a + 2.0*M_PI * k;
      }

      /* if cos(l(x)) > u(s), we are looking for k minimal s.t. pi - a + 2k*pi > l(x) where a = acos(u(s))
       * then the new lower bound is pi - a + 2k*pi
       */
      else if( SCIPisGT(scip, COS(newinf), interval.sup) )
      {
         SCIP_Real a = ACOS(interval.sup);
         int k = (int) ceil((newinf + a) / (2.0*M_PI) - 0.5);
         newinf = M_PI * (2.0*k + 1.0) - a;
      }
   }

   if( !SCIPisInfinity(scip, newsup) )
   {
      /* if cos(u(x)) > u(s), we are looking for k minimal s.t. a + 2k*pi > u(x) - 2*pi where a = acos(u(s))
       * then the new upper bound is a + 2k*pi
       */
      if ( SCIPisGT(scip, COS(newsup), interval.sup) )
      {
         SCIP_Real a = ACOS(interval.sup);
         int k = (int) ceil((newsup - a ) / (2.0*M_PI)) - 1;
         newsup = a + 2.0*M_PI * k;
      }

      /* if cos(u(x)) < l(s), we are looking for k minimal s.t. pi - a + 2k*pi > l(x) - 2*pi where a = acos(l(s))
       * then the new upper bound is pi - a + 2k*pi
       */
      if( SCIPisLT(scip, COS(newsup), interval.inf) )
      {
         SCIP_Real a = ACOS(interval.inf);
         int k = (int) ceil((newsup + a) / (2*M_PI) - 0.5) - 1;
         newsup = M_PI * (2.0*k + 1.0) - a;
      }
   }

   assert(newinf >= childbound.inf);
   assert(newsup <= childbound.sup);
   assert(newinf <= newsup);
   assert(SCIPisGE(scip, COS(newinf), interval.inf));
   assert(SCIPisLE(scip, COS(newinf), interval.sup));
   assert(SCIPisGE(scip, COS(newsup), interval.inf));
   assert(SCIPisLE(scip, COS(newsup), interval.sup));

   SCIPintervalSetBounds(&childbound, newinf, newsup);

   /* try to tighten the bounds of the child node */
   SCIP_CALL( SCIPtightenConsExprExprInterval(scip, child, childbound, force, infeasible, nreductions) );

   return SCIP_OKAY;
}

/** cos hash callback */
static
SCIP_DECL_CONSEXPR_EXPRHASH(hashCos)
{  /*lint --e{715}*/
   unsigned int childhash;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(expr2key != NULL);
   assert(hashkey != NULL);

   *hashkey = EXPRHDLR_HASHKEY;

   assert(SCIPhashmapExists(expr2key, (void*) SCIPgetConsExprExprChildren(expr)[0]));
   childhash = (unsigned int)(size_t) SCIPhashmapGetImage(expr2key, SCIPgetConsExprExprChildren(expr)[0]);

   *hashkey ^= childhash;

   return SCIP_OKAY;
}

/** creates the handler for cos expressions and includes it into the expression constraint handler */
SCIP_RETCODE SCIPincludeConsExprExprHdlrCos(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;

   /* include expression handler */
   SCIP_CALL( SCIPincludeConsExprExprHdlrBasic(scip, consexprhdlr, &exprhdlr, EXPRHDLR_NAME, EXPRHDLR_DESC,
         EXPRHDLR_PRECEDENCE, evalCos, NULL) );
   assert(exprhdlr != NULL);

   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeHdlr(scip, consexprhdlr, exprhdlr, copyhdlrCos, NULL) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeData(scip, consexprhdlr, exprhdlr, copydataCos, freedataCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrSimplify(scip, consexprhdlr, exprhdlr, simplifyCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrPrint(scip, consexprhdlr, exprhdlr, printCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrParse(scip, consexprhdlr, exprhdlr, parseCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrIntEval(scip, consexprhdlr, exprhdlr, intevalCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrInitSepa(scip, consexprhdlr, exprhdlr, initSepaCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrSepa(scip, consexprhdlr, exprhdlr, sepaCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrReverseProp(scip, consexprhdlr, exprhdlr, reversepropCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrHash(scip, consexprhdlr, exprhdlr, hashCos) );
   SCIP_CALL( SCIPsetConsExprExprHdlrBwdiff(scip, consexprhdlr, exprhdlr, bwdiffCos) );

   return SCIP_OKAY;
}

/** creates a cos expression */
SCIP_RETCODE SCIPcreateConsExprExprCos(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR**  expr,               /**< pointer where to store expression */
   SCIP_CONSEXPR_EXPR*   child               /**< single child */
   )
{
   assert(expr != NULL);
   assert(child != NULL);
   assert(SCIPfindConsExprExprHdlr(consexprhdlr, EXPRHDLR_NAME) != NULL);

   SCIP_CALL( SCIPcreateConsExprExpr(scip, expr, SCIPfindConsExprExprHdlr(consexprhdlr, EXPRHDLR_NAME), NULL, 1,
         &child) );

   return SCIP_OKAY;
}
