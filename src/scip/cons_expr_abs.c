/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2016 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_expr_abs.c
 * @brief  absolute expression handler
 * @author Stefan Vigerske
 * @author Benjamin Mueller
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <string.h>

#include "scip/cons_expr_value.h"
#include "scip/cons_expr_abs.h"

#define SCIP_PRIVATE_ROWPREP
#include "scip/cons_quadratic.h"

#define EXPRHDLR_NAME         "abs"
#define EXPRHDLR_DESC         "absolute expression"
#define EXPRHDLR_PRECEDENCE  70000
#define EXPRHDLR_HASHKEY     SCIPcalcFibHash(7187.0)

/*
 * Data structures
 */

struct SCIP_ConsExpr_ExprData
{
   SCIP_ROW*  rowneg;  /**< left tangent z >= -x */
   SCIP_ROW*  rowpos;  /**< right tangent z <= x */
};

/*
 * Local methods
 */


/*
 * Callback methods of expression handler
 */

/** simplifies an abs expression.
 * Evaluates the absolute value function when its child is a value expression
 * TODO: abs(*) = * if * >= 0 or - * if * < 0
 */
static
SCIP_DECL_CONSEXPR_EXPRSIMPLIFY(simplifyAbs)
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
      SCIP_CALL( SCIPcreateConsExprExprValue(scip, conshdlr, simplifiedexpr, REALABS(SCIPgetConsExprExprValueValue(child))) );
   }
   else
   {
      *simplifiedexpr = expr;

      /* we have to capture it, since it must simulate a "normal" simplified call in which a new expression is created */
      SCIPcaptureConsExprExpr(*simplifiedexpr);
   }

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRCOPYHDLR(copyhdlrAbs)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPincludeConsExprExprHdlrAbs(scip, consexprhdlr) );
   *valid = TRUE;

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRCOPYDATA(copydataAbs)
{  /*lint --e{715}*/
   assert(targetscip != NULL);
   assert(targetexprdata != NULL);

   SCIP_CALL( SCIPallocBlockMemory(targetscip, targetexprdata) );
   BMSclearMemory(*targetexprdata);

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRFREEDATA(freedataAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRDATA* exprdata;

   assert(expr != NULL);

   exprdata = SCIPgetConsExprExprData(expr);
   assert(exprdata != NULL);

   SCIPfreeBlockMemory(scip, &exprdata);
   SCIPsetConsExprExprData(expr, NULL);

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRPRINT(printAbs)
{  /*lint --e{715}*/
   assert(expr != NULL);

   switch( stage )
   {
      case SCIP_CONSEXPREXPRWALK_ENTEREXPR :
      {
         /* print function with opening parenthesis */
         SCIPinfoMessage(scip, file, "abs(");
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

static
SCIP_DECL_CONSEXPR_EXPRPARSE(parseAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* childexpr;

   assert(expr != NULL);

   /* parse child expression from remaining string */
   SCIP_CALL( SCIPparseConsExprExpr(scip, consexprhdlr, string, endstring, &childexpr) );
   assert(childexpr != NULL);

   /* create absolute expression */
   SCIP_CALL( SCIPcreateConsExprExprAbs(scip, consexprhdlr, expr, childexpr) );
   assert(*expr != NULL);

   /* release child expression since it has been captured by the absolute expression */
   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &childexpr) );

   *success = TRUE;

   return SCIP_OKAY;
}

/** expression point evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPREVAL(evalAbs)
{  /*lint --e{715}*/
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(SCIPgetConsExprExprValue(SCIPgetConsExprExprChildren(expr)[0]) != SCIP_INVALID); /*lint !e777*/

   *val = REALABS(SCIPgetConsExprExprValue(SCIPgetConsExprExprChildren(expr)[0]));

   return SCIP_OKAY;
}


/** expression derivative evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRBWDIFF(bwdiffAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;

   assert(expr != NULL);
   assert(SCIPgetConsExprExprData(expr) != NULL);
   assert(childidx == 0);
   assert(SCIPgetConsExprExprValue(expr) != SCIP_INVALID); /*lint !e777*/

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(child)), "val") != 0);

   *val = (SCIPgetConsExprExprValue(child) >= 0.0) ? 1.0 : -1.0;

   return SCIP_OKAY;
}

/** expression interval evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRINTEVAL(intevalAbs)
{  /*lint --e{715}*/
   SCIP_INTERVAL childinterval;

   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   childinterval = SCIPgetConsExprExprInterval(SCIPgetConsExprExprChildren(expr)[0]);
   assert(!SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, childinterval));

   SCIPintervalAbs(SCIP_INTERVAL_INFINITY, interval, childinterval);

   return SCIP_OKAY;
}

/** computes both tangent underestimates and secant */
static
SCIP_RETCODE computeCutsAbs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR*   expr,               /**< absolute expression */
   SCIP_Bool             overestimate,       /**< overestimate the absolute expression? */
   SCIP_Bool             underestimate,      /**< underestimate the absolute expression? */
   SCIP_ROW**            rowneg,             /**< buffer to store first tangent (might be NULL) */
   SCIP_ROW**            rowpos,             /**< buffer to store second tangent (might be NULL) */
   SCIP_ROW**            secant              /**< buffer to store secant (might be NULL) */
   )
{
   SCIP_VAR* x;
   SCIP_VAR* z;
   SCIP_VAR* vars[2];
   SCIP_Real coefs[2];
   char name[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), EXPRHDLR_NAME) == 0);

   x = SCIPgetConsExprExprAuxVar(SCIPgetConsExprExprChildren(expr)[0]);
   z = SCIPgetConsExprExprAuxVar(expr);
   assert(x != NULL);
   assert(z != NULL);
   /* z = abs(x) */

   vars[0] = z;
   vars[1] = x;
   coefs[0] = -1.0;

   /* compute left tangent -z -x <= 0 */
   if( rowneg != NULL && underestimate )
   {
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "abs_neg_%s", SCIPvarGetName(x));
      coefs[1] = -1.0;
      SCIP_CALL( SCIPcreateEmptyRowCons(scip, rowneg, conshdlr, name, -SCIPinfinity(scip), 0.0, FALSE, FALSE, FALSE) );
      SCIP_CALL( SCIPaddVarsToRow(scip, *rowneg, 2, vars, coefs) );
   }

   /* compute right tangent -z +x <= 0 */
   if( rowpos != NULL && underestimate )
   {
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "abs_pos_%s", SCIPvarGetName(x));
      coefs[1] = 1.0;
      SCIP_CALL( SCIPcreateEmptyRowCons(scip, rowpos, conshdlr, name, -SCIPinfinity(scip), 0.0, FALSE, FALSE, FALSE) );
      SCIP_CALL( SCIPaddVarsToRow(scip, *rowpos, 2, vars, coefs) );
   }

   /* compute secant */
   if( secant != NULL && overestimate )
   {
      SCIP_Real lb;
      SCIP_Real ub;

      *secant = NULL;
      lb = SCIPvarGetLbLocal(x);
      ub = SCIPvarGetUbLocal(x);

      /* it does not make sense to add a cut if child variable is unbounded or fixed */
      if( !SCIPisInfinity(scip, -lb) && !SCIPisInfinity(scip, ub) && !SCIPisEQ(scip, lb, ub) )
      {
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "abs_secant_%s", SCIPvarGetName(x));

         if( !SCIPisPositive(scip, ub) )
         {
            /* z = -x, so add -z-x >= 0 here (-z-x <= 0 is the underestimator that is added above) */
            coefs[1] = -1.0;
            SCIP_CALL( SCIPcreateEmptyRowCons(scip, secant, conshdlr, name, 0.0, SCIPinfinity(scip), TRUE, FALSE, FALSE) );
            SCIP_CALL( SCIPaddVarsToRow(scip, *secant, 2, vars, coefs) );
         }
         else if( !SCIPisNegative(scip, lb) )
         {
            /* z =  x, so add -z+x >= 0 here (-z+x <= 0 is the underestimator that is added above) */
            coefs[1] =  1.0;
            SCIP_CALL( SCIPcreateEmptyRowCons(scip, secant, conshdlr, name, 0.0, SCIPinfinity(scip), TRUE, FALSE, FALSE) );
            SCIP_CALL( SCIPaddVarsToRow(scip, *secant, 2, vars, coefs) );
         }
         else
         {
            /* z = abs(x), x still has mixed sign */
            SCIP_Real alpha;
            SCIP_ROWPREP* rowprep;
            SCIP_Bool success;

            /* let alpha = (|ub|-|lb|) / (ub-lb) then the resulting secant looks like
             *
             * z - |ub| <= alpha * (x - ub)  <=> alpha * ub - |ub| <= -z + alpha * x
             */
            alpha = (REALABS(ub) - REALABS(lb)) / (ub - lb);

            /* create row preparation */
            SCIP_CALL( SCIPcreateRowprep(scip, &rowprep, SCIP_SIDETYPE_LEFT, TRUE) );
            SCIPaddRowprepSide(rowprep, alpha * ub - REALABS(ub));
            coefs[1] = alpha;
            SCIP_CALL( SCIPaddRowprepTerms(scip, rowprep, 2, vars, coefs) );

            /* cleanup coefficient and side, esp treat epsilon to integral values; don't consider scaling up here */
            SCIP_CALL( SCIPcleanupRowprep(scip, rowprep, NULL, SCIP_CONSEXPR_CUTMAXRANGE, 0.0, NULL, &success) );

            /* if rowprep is good, then create SCIP_ROW */
            if( success )
            {
               memcpy(rowprep->name, name, (unsigned long)SCIP_MAXSTRLEN);
               SCIP_CALL( SCIPgetRowprepRowCons(scip, secant, rowprep, conshdlr) );
            }

            SCIPfreeRowprep(scip, &rowprep);
         }
      }
   }

   return SCIP_OKAY;
}

/** expression separation initialization callback */
static
SCIP_DECL_CONSEXPR_EXPRINITSEPA(initSepaAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRDATA* exprdata;
   SCIP_ROW* secant;

   exprdata = SCIPgetConsExprExprData(expr);
   assert(exprdata != NULL);
   assert(exprdata->rowneg == NULL);
   assert(exprdata->rowpos == NULL);

   *infeasible = FALSE;
   secant = NULL;

   /* compute initial cuts; do no store the secant in the expression data */
   SCIP_CALL( computeCutsAbs(scip, conshdlr, expr, overestimate, underestimate, &exprdata->rowneg, &exprdata->rowpos,
      &secant) );
   assert(exprdata->rowneg != NULL || !underestimate);
   assert(exprdata->rowpos != NULL || !underestimate);

   /* add cuts */
   if( exprdata->rowneg != NULL )
   {
      SCIP_CALL( SCIPaddRow(scip, exprdata->rowneg, FALSE, infeasible) );
   }

   if( !*infeasible && exprdata->rowpos != NULL )
   {
      SCIP_CALL( SCIPaddRow(scip, exprdata->rowpos, FALSE, infeasible) );
   }

   /* it might happen that we could not compute a secant (because of fixed or unbounded variables) */
   if( !*infeasible && secant != NULL )
   {
      SCIP_CALL( SCIPaddRow(scip, secant, FALSE, infeasible) );
   }

   /* release secant */
   if( secant != NULL )
   {
      SCIP_CALL( SCIPreleaseRow(scip, &secant) );
   }
   assert(secant == NULL);

   return SCIP_OKAY;
}

/** expression separation deinitialization callback */
static
SCIP_DECL_CONSEXPR_EXPREXITSEPA(exitSepaAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRDATA* exprdata;

   exprdata = SCIPgetConsExprExprData(expr);
   assert(exprdata != NULL);

   if( exprdata->rowneg != NULL )
   {
      SCIP_CALL( SCIPreleaseRow(scip, &exprdata->rowneg) );
   }

   if( exprdata->rowpos != NULL )
   {
      SCIP_CALL( SCIPreleaseRow(scip, &exprdata->rowpos) );
   }

   assert(exprdata->rowneg == NULL);
   assert(exprdata->rowpos == NULL);

   return SCIP_OKAY;
}


/** expression separation callback */
static
SCIP_DECL_CONSEXPR_EXPRSEPA(sepaAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRDATA* exprdata;
   SCIP_ROW* rows[3] = {NULL, NULL, NULL};
   SCIP_Real violation;
   SCIP_Bool infeasible;
   int i;

   exprdata = SCIPgetConsExprExprData(expr);
   assert(exprdata != NULL);

   infeasible = FALSE;
   *ncuts = 0;
   *result = SCIP_DIDNOTFIND;

   /* create all cuts that might be relevant */
   if( !overestimate )
   {
      /* create tangents if it not happened so far (might be possible if the constraint is not 'initial') */
      if( exprdata->rowneg == NULL )
      {
         SCIP_CALL( computeCutsAbs(scip, conshdlr, expr, FALSE, TRUE, &exprdata->rowneg, NULL, NULL) );
      }
      if( exprdata->rowpos == NULL )
      {
         SCIP_CALL( computeCutsAbs(scip, conshdlr, expr, FALSE, TRUE, NULL, &exprdata->rowpos, NULL) );
      }
   }
   else
   {
      SCIP_CALL( computeCutsAbs(scip, conshdlr, expr, TRUE, FALSE, NULL, NULL, &rows[2]) );

      /* check whether violation >= mincutviolation */
      if( rows[2] != NULL && -SCIPgetRowSolFeasibility(scip, rows[2], sol) < mincutviolation )
      {
         SCIP_CALL( SCIPreleaseRow(scip, &rows[2]) );
      }
   }

   assert(exprdata->rowneg != NULL || overestimate);
   assert(exprdata->rowpos != NULL || overestimate);

   rows[0] = exprdata->rowneg;
   rows[1] = exprdata->rowpos;

   for( i = 0; i < 3; ++i )
   {
      if( rows[i] == NULL || SCIProwIsInLP(rows[i]) )
         continue;

      violation = -SCIPgetRowSolFeasibility(scip, rows[i], sol);
      if( SCIPisGE(scip, violation, mincutviolation) )
      {
         SCIP_CALL( SCIPaddRow(scip, rows[i], FALSE, &infeasible) );

         if( infeasible )
         {
            *result = SCIP_CUTOFF;
            break;
         }
         else
         {
            *result = SCIP_SEPARATED;
            ++*ncuts;
         }
      }
   }

   /* release the secant */
   if( rows[2] != NULL )
   {
      SCIP_CALL( SCIPreleaseRow(scip, &rows[2]) );
   }

   return SCIP_OKAY;
}

/** expression reverse propagation callback */
static
SCIP_DECL_CONSEXPR_REVERSEPROP(reversepropAbs)
{  /*lint --e{715}*/
   SCIP_INTERVAL childbounds;
   SCIP_INTERVAL left;
   SCIP_INTERVAL right;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(nreductions != NULL);
   assert(SCIPintervalGetInf(SCIPgetConsExprExprInterval(expr)) >= 0.0);

   *nreductions = 0;

   /* abs(x) in I -> x \in (-I \cup I) \cap bounds(x) */
   right = SCIPgetConsExprExprInterval(expr);  /* I */
   SCIPintervalSetBounds(&left, -right.sup, -right.inf); /* -I */

   childbounds = SCIPgetConsExprExprInterval(SCIPgetConsExprExprChildren(expr)[0]);
   SCIPintervalIntersect(&left, left, childbounds);    /* -I \cap bounds(x), could become empty */
   SCIPintervalIntersect(&right, right, childbounds);  /*  I \cap bounds(x), could become empty */
   /* compute smallest interval containing (-I \cap bounds(x)) \cup (I \cap bounds(x)) = (-I \cup I) \cap bounds(x)
    * this works also if left or right is empty
    */
   SCIPintervalUnify(&childbounds, left, right);

   /* try to tighten the bounds of the child node */
   SCIP_CALL( SCIPtightenConsExprExprInterval(scip, SCIPgetConsExprExprChildren(expr)[0], childbounds, force, reversepropqueue, infeasible,
         nreductions) );

   return SCIP_OKAY;
}

/** expression hash callback */
static
SCIP_DECL_CONSEXPR_EXPRHASH(hashAbs)
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

/** expression curvature detection callback */
static
SCIP_DECL_CONSEXPR_EXPRCURVATURE(curvatureAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_EXPRCURV childcurv;
   SCIP_Real childinf;
   SCIP_Real childsup;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(curvature != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);

   childcurv = SCIPgetConsExprExprCurvature(child);
   childinf = SCIPintervalGetInf(SCIPgetConsExprExprInterval(child));
   childsup = SCIPintervalGetSup(SCIPgetConsExprExprInterval(child));

   *curvature = SCIP_EXPRCURV_UNKNOWN;

   /* TODO do we need to consider the cases where childinf >= 0 or childsup <= 0.0 holds? */
   switch( childcurv )
   {
      case SCIP_EXPRCURV_UNKNOWN:
         *curvature = SCIP_EXPRCURV_UNKNOWN;
         break;

      case SCIP_EXPRCURV_CONVEX:
         if( childinf >= 0.0 )
            *curvature = SCIP_EXPRCURV_CONVEX;
         else if( childsup <= 0.0 )
            *curvature = SCIP_EXPRCURV_CONCAVE;
         break;

      case SCIP_EXPRCURV_CONCAVE:
         if( childsup <= 0.0 )
            *curvature = SCIP_EXPRCURV_CONVEX;
         else if( childinf >= 0.0 )
            *curvature = SCIP_EXPRCURV_CONCAVE;
         break;

      case SCIP_EXPRCURV_LINEAR:
         *curvature = SCIP_EXPRCURV_CONVEX;
         break;
   }

   return SCIP_OKAY;
}

/** expression monotonicity detection callback */
static
SCIP_DECL_CONSEXPR_EXPRMONOTONICITY(monotonicityAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_Real childinf;
   SCIP_Real childsup;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(result != NULL);
   assert(childidx == 0);

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);

   childinf = SCIPintervalGetInf(SCIPgetConsExprExprInterval(child));
   childsup = SCIPintervalGetSup(SCIPgetConsExprExprInterval(child));

   if( childsup <= 0.0 )
      *result = SCIP_MONOTONE_DEC;
   else if( childinf >= 0.0 )
      *result = SCIP_MONOTONE_INC;
   else
      *result = SCIP_MONOTONE_UNKNOWN;

   return SCIP_OKAY;
}

/** expression integrality detection callback */
static
SCIP_DECL_CONSEXPR_EXPRINTEGRALITY(integralityAbs)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(isintegral != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);

   *isintegral = SCIPisConsExprExprIntegral(child);

   return SCIP_OKAY;
}


/** creates the handler for absolute expression and includes it into the expression constraint handler */
SCIP_RETCODE SCIPincludeConsExprExprHdlrAbs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;

   SCIP_CALL( SCIPincludeConsExprExprHdlrBasic(scip, consexprhdlr, &exprhdlr, EXPRHDLR_NAME, EXPRHDLR_DESC,
         EXPRHDLR_PRECEDENCE, evalAbs, NULL) );
   assert(exprhdlr != NULL);

   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeHdlr(scip, consexprhdlr, exprhdlr, copyhdlrAbs, NULL) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeData(scip, consexprhdlr, exprhdlr, copydataAbs, freedataAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrSimplify(scip, consexprhdlr, exprhdlr, simplifyAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrPrint(scip, consexprhdlr, exprhdlr, printAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrParse(scip, consexprhdlr, exprhdlr, parseAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrIntEval(scip, consexprhdlr, exprhdlr, intevalAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrSepa(scip, consexprhdlr, exprhdlr, initSepaAbs, exitSepaAbs, sepaAbs, NULL) );
   SCIP_CALL( SCIPsetConsExprExprHdlrHash(scip, consexprhdlr, exprhdlr, hashAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrReverseProp(scip, consexprhdlr, exprhdlr, reversepropAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrBwdiff(scip, consexprhdlr, exprhdlr, bwdiffAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCurvature(scip, consexprhdlr, exprhdlr, curvatureAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrMonotonicity(scip, consexprhdlr, exprhdlr, monotonicityAbs) );
   SCIP_CALL( SCIPsetConsExprExprHdlrIntegrality(scip, consexprhdlr, exprhdlr, integralityAbs) );

   return SCIP_OKAY;
}

/** creates an absolute expression */
SCIP_RETCODE SCIPcreateConsExprExprAbs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR**  expr,               /**< pointer where to store expression */
   SCIP_CONSEXPR_EXPR*   child               /**< single child */
   )
{
   SCIP_CONSEXPR_EXPRDATA* exprdata;

   assert(expr != NULL);
   assert(child != NULL);
   assert(SCIPfindConsExprExprHdlr(consexprhdlr, EXPRHDLR_NAME) != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, &exprdata) );
   assert(exprdata != NULL);

   BMSclearMemory(exprdata);

   SCIP_CALL( SCIPcreateConsExprExpr(scip, expr, SCIPfindConsExprExprHdlr(consexprhdlr, EXPRHDLR_NAME), exprdata, 1, &child) );

   return SCIP_OKAY;
}
