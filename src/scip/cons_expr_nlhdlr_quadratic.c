/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2018 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scip.zib.de.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_expr_nlhdlr_quadratic.c
 * @brief  nonlinear handler to handle quadratic expressions
 * @author Felipe Serrano
 *
 * Some definitions:
 * - a BILINEXPRTERM is a product of two expressions
 * - a SCIP_QUADEXPRTERM stores an expression expr that is known to appear in a nonlinear, quadratic term, that is
 *   expr^2 or expr * other_expr. It stores its sqrcoef (that can be 0), its linear coef and all the bilinear expression
 *   terms in which expr appears.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <string.h>

#include "scip/cons_expr_nlhdlr_quadratic.h"
#include "scip/cons_expr_pow.h"
#include "scip/cons_expr_sum.h"
#include "scip/cons_expr_var.h"
#include "scip/cons_expr_product.h"

#include "nlpi/nlpi_ipopt.h" /* for LAPACK */

/* fundamental nonlinear handler properties */
#define NLHDLR_NAME         "quadratic"
#define NLHDLR_DESC         "handler for quadratic expressions"
#define NLHDLR_PRIORITY     100

/*
 * Data structures
 */

/** nonlinear handler expression data */
struct SCIP_ConsExpr_NlhdlrExprData
{
   int                   nlinexprs;          /**< number of expressions that appear linearly */
   SCIP_CONSEXPR_EXPR**  linexprs;           /**< expressions that appear linearly */
   SCIP_Real*            lincoefs;           /**< coefficients of expressions that appear linearly */

   int                   nquadexprs;         /**< number of expressions in quadratic terms */
   SCIP_QUADEXPRTERM*    quadexprterms;      /**< array with quadratic expression terms */

   int                   nbilinexprterms;    /**< number of bilinear expressions terms */
   SCIP_BILINEXPRTERM*   bilinexprterms;     /**< bilinear expression terms array */

   SCIP_EXPRCURV         curvature;          /**< curvature of the quadratic representation of the expression */

   SCIP_INTERVAL         linactivity;        /**< activity of linear part */

   /* activities of quadratic parts as defined in nlhdlrIntervalQuadratic */
   SCIP_Real             minquadfiniteact;   /**< minimum activity of quadratic part where only terms with finite min
                                               activity contribute */
   SCIP_Real             maxquadfiniteact;   /**< maximum activity of quadratic part where only terms with finite max
                                               activity contribute */
   int                   nneginfinityquadact;/**< number of quadratic terms contributing -infinity to activity */
   int                   nposinfinityquadact;/**< number of quadratic terms contributing +infinity to activity */
   SCIP_INTERVAL*        quadactivities;     /**< activity of each quadratic term as defined in nlhdlrIntervalQuadratic */
};

/*
 * static methods
 */


/** frees nlhdlrexprdata structure */
static
void freeNlhdlrExprData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata /**< nlhdlr expression data */
   )
{
   int i;

   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->linexprs), nlhdlrexprdata->nlinexprs);
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->lincoefs), nlhdlrexprdata->nlinexprs);
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->bilinexprterms), nlhdlrexprdata->nbilinexprterms);

   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->quadactivities), nlhdlrexprdata->nquadexprs);

   for( i = 0; i < nlhdlrexprdata->nquadexprs; ++i )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->quadexprterms[i].adjbilin),
            nlhdlrexprdata->quadexprterms[i].adjbilinsize);
   }
   SCIPfreeBlockMemoryArrayNull(scip, &(nlhdlrexprdata->quadexprterms), nlhdlrexprdata->nquadexprs);
}

/** first time seen quadratically and
 * seen before linearly --> --nlinterms; assign 2; ++nquadterms
 * not seen before linearly --> assing 1; ++nquadterms
 *
 * seen before --> assign += 1
 */
static
SCIP_RETCODE processQuadraticExpr(
   SCIP_CONSEXPR_EXPR*   expr,               /**< the expression */
   SCIP_HASHMAP*         seenexpr,           /**< hash map */
   SCIP_Bool*            proper,             /**< buffer to store whether this expr makes the quadratic proper */
   int*                  nquadterms,         /**< number of quadratic terms */
   int*                  nlinterms           /**< number of linear terms */
   )
{
   if( SCIPhashmapExists(seenexpr, (void *)expr) )
   {
      if( SCIPhashmapGetImageInt(seenexpr, (void *)expr) < 0 )
      {
         /* only seen linearly before */
         assert(SCIPhashmapGetImageInt(seenexpr, (void *)expr) == -1);

         --(*nlinterms);
         ++(*nquadterms);
         SCIP_CALL( SCIPhashmapSetImageInt(seenexpr, (void *)expr, 2) );
      }
      else
      {
         assert(SCIPhashmapGetImageInt(seenexpr, (void *)expr) > 0);
         SCIP_CALL( SCIPhashmapSetImageInt(seenexpr, (void *)expr,
                  SCIPhashmapGetImageInt(seenexpr, (void *)expr) + 1) );
      }
      *proper = TRUE;
   }
   else
   {
      ++(*nquadterms);
      SCIP_CALL( SCIPhashmapInsertInt(seenexpr, (void *)expr, 1) );
   }

   return SCIP_OKAY;
}

/** Checks the curvature of the quadratic function, x^T Q x + b^T x stored in nlhdlrexprdata; for this, it builds the
 * matrix Q and computes its eigenvalues using LAPACK; if Q is
 * - semidefinite positive -> provided is set to sepaunder
 * - semidefinite negative -> provided is set to sepaover
 * - otherwise -> provided is set to none
 */
/* TODO: make more simple test; like diagonal entries don't change sign, etc */
static
SCIP_RETCODE checkCurvature(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata /**< nlhdlr expression data */
   )
{
   SCIP_HASHMAP* expr2matrix;
   double* matrix;
   double* alleigval;
   int nvars;
   int nn;
   int n;
   int i;

   nlhdlrexprdata->curvature = SCIP_EXPRCURV_UNKNOWN;

   n  = nlhdlrexprdata->nquadexprs;
   nn = n * n;

   /* do not check curvature if nn is too large */
   if( nn < 0 || (unsigned) (int) nn > UINT_MAX / sizeof(SCIP_Real) )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "nlhdlr_quadratic - number of quadratic variables is too large (%d) to check the curvature; will not handle this expression\n", n);

      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPallocBufferArray(scip, &alleigval, n) );
   SCIP_CALL( SCIPallocClearBufferArray(scip, &matrix, nn) );

   SCIP_CALL( SCIPhashmapCreate(&expr2matrix, SCIPblkmem(scip), n) );

   /* fill matrix's diagonal */
   nvars = 0;
   for( i = 0; i < n; ++i )
   {
      SCIP_QUADEXPRTERM quadexprterm;

      quadexprterm = nlhdlrexprdata->quadexprterms[i];

      assert(!SCIPhashmapExists(expr2matrix, (void*)quadexprterm.expr));

      if( quadexprterm.sqrcoef == 0.0 )
      {
         assert(quadexprterm.nadjbilin > 0);
         /* SCIPdebugMsg(scip, "var <%s> appears in bilinear term but is not squared --> indefinite quadratic\n", SCIPvarGetName(quadexprterm.var)); */
         goto CLEANUP;
      }

      matrix[nvars * n + nvars] = quadexprterm.sqrcoef;

      /* remember row of variable in matrix */
      SCIP_CALL( SCIPhashmapInsert(expr2matrix, (void *)quadexprterm.expr, (void *)(size_t)nvars) );
      nvars++;
   }

   /* fill matrix's upper-diagonal */
   for( i = 0; i < nlhdlrexprdata->nbilinexprterms; ++i )
   {
      SCIP_BILINEXPRTERM bilinexprterm;
      int col;
      int row;

      bilinexprterm = nlhdlrexprdata->bilinexprterms[i];

      assert(SCIPhashmapExists(expr2matrix, (void*)bilinexprterm.expr1));
      assert(SCIPhashmapExists(expr2matrix, (void*)bilinexprterm.expr2));
      row = (int)(size_t)SCIPhashmapGetImage(expr2matrix, bilinexprterm.expr1);
      col = (int)(size_t)SCIPhashmapGetImage(expr2matrix, bilinexprterm.expr2);

      assert(row != col);

      if( row < col )
         matrix[row * n + col] = bilinexprterm.coef / 2.0;
      else
         matrix[col * n + row] = bilinexprterm.coef / 2.0;
   }

   /* compute eigenvalues */
   if( LapackDsyev(FALSE, n, matrix, alleigval) != SCIP_OKAY )
   {
      SCIPwarningMessage(scip, "Failed to compute eigenvalues of quadratic coefficient matrix --> don't know curvature\n");
      goto CLEANUP;
   }

   /* check convexity */
   if( !SCIPisNegative(scip, alleigval[0]) )
   {
      nlhdlrexprdata->curvature = SCIP_EXPRCURV_CONVEX;
   }
   else if( !SCIPisPositive(scip, alleigval[n-1]) )
   {
      nlhdlrexprdata->curvature = SCIP_EXPRCURV_CONCAVE;
   }

CLEANUP:
   SCIPhashmapFree(&expr2matrix);
   SCIPfreeBufferArray(scip, &matrix);
   SCIPfreeBufferArray(scip, &alleigval);

   return SCIP_OKAY;
}


/** creates auxiliary variable when necessary */
static
SCIP_RETCODE createAuxVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< expr conshdlr */
   SCIP_CONSEXPR_EXPR*   expr,               /**< expression to add to quadratic terms */
   SCIP_Bool*            originalvar         /**< set it to false when expression is not var */
   )
{
   if( SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrVar(conshdlr) )
      return SCIP_OKAY;

   *originalvar = FALSE;
   SCIP_CALL( SCIPcreateConsExprExprAuxVar(scip, conshdlr, expr, NULL) );

   return SCIP_OKAY;
}

/** solves a quadratic equation \f$ a expr^2 + b expr \in rhs \f$ (with b an interval) and reduces bounds on expr or
 * deduces infeasibility if possible; expr is quadexpr.expr
 */
static
SCIP_RETCODE propagateBoundsQuadExpr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_QUADEXPRTERM     quadexpr,           /**< quadratic expression to propagate */
   SCIP_INTERVAL         b,                  /**< interval acting as linear coefficient */
   SCIP_INTERVAL         rhs,                /**< interval acting as rhs */
   SCIP_QUEUE*           reversepropqueue,   /**< queue used in reverse prop, pass to SCIPtightenConsExprExprInterval */
   SCIP_Bool*            infeasible,         /**< buffer to store if propagation produced infeasibility */
   int*                  nreductions,        /**< buffer to store the number of interval reductions */
   SCIP_Bool             force               /**< to force tightening */
   )
{
   SCIP_INTERVAL a;
   SCIP_INTERVAL newrange;

   assert(scip != NULL);
   assert(infeasible != NULL);
   assert(nreductions != NULL);

#ifdef DEBUG_PROP
   SCIPinfoMessage(scip, NULL, "Propagating <expr> by solving a <expr>^2 + b <expr> in rhs, where <expr> is: ");
   SCIP_CALL( SCIPprintConsExprExpr(scip, SCIPfindConshdlr(scip, "expr"), quadexpr.expr, NULL) );
   SCIPinfoMessage(scip, NULL, "\n");
   SCIPinfoMessage(scip, NULL, "expr in [%g, %g], a = %g, b = [%g, %g] and rhs = [%g, %g]\n",
         SCIPintervalGetInf(SCIPgetConsExprExprInterval(quadexpr.expr)),
         SCIPintervalGetSup(SCIPgetConsExprExprInterval(quadexpr.expr)), quadexpr.sqrcoef, b.inf, b.sup,
         rhs.inf, rhs.sup);
#endif

   /* compute solution of a*x^2 + b*x \in rhs */
   SCIPintervalSet(&a, quadexpr.sqrcoef);
   SCIPintervalSolveUnivariateQuadExpression(SCIP_INTERVAL_INFINITY, &newrange, a, b, rhs, SCIPgetConsExprExprInterval(quadexpr.expr));

#ifdef DEBUG_PROP
   SCIPinfoMessage(scip, NULL, "Solution [%g, %g]\n", newrange.inf, newrange.sup);
#endif

   SCIP_CALL( SCIPtightenConsExprExprInterval(scip, quadexpr.expr, newrange, force, reversepropqueue, infeasible,
            nreductions) );

   return SCIP_OKAY;
}

/*
 * Callback methods of nonlinear handler
 */

/** callback to free expression specific data */
static
SCIP_DECL_CONSEXPR_NLHDLRFREEEXPRDATA(nlhdlrfreeExprDataQuadratic)
{  /*lint --e{715}*/
   freeNlhdlrExprData(scip, *nlhdlrexprdata);
   SCIPfreeBlockMemory(scip, nlhdlrexprdata);

   return SCIP_OKAY;
}

/** callback to detect structure in expression tree
 *
 * A term is quadratic if:
 * - It is a product expression of two expressions
 * - It is power expression of an expression with exponent 2.0
 *
 * A proper quadratic expression (i.e the only quadratic expressions that can be handled by this nlhdlr) is a sum
 * expression such that there is at least one expr that appears at least twice (because of simplification,
 * this means it appears in a quadratic terms and somewhere else).
 * For example: x^2 + y^2 is not a proper quadratic expression; x^2 + x is proper quadratic expression;
 * x^2 + x * y is also a proper quadratic expression
 *
 * For propagation, we store the quadratic in our data structure in the following way:
 * We count how often a variable appears. Then, in a bilinear product, expr_i * expr_j,
 * we store it as expr_i * expr_j if and only if # expr_i appears >= # expr_j appears.
 *
 * @note:
 * - the expression needs to be simplified (in particular, it is assumed to be sorted)
 * - common subexpressions are also assumed to have been identified, the hashing will fail otherwise!
 *
 * Sorted implies that:
 *  - expr < expr^2: bases are the same, but exponent 1 < 2
 *  - expr < expr * other_expr: u*v < w holds if and only if v < w (OR8), but here w = u < v, since expr comes before
 *  other_expr in the product
 *  - expr < other_expr * expr: u*v < w holds if and only if v < w (OR8), but here v = w
 *
 *  Thus, if we see somebody twice, it is a proper quadratic.
 *
 * It also implies that
 *  - expr^2 < expr * other_expr
 *  - other_expr * expr < expr^2
 *
 * It also implies that x^-2 < x^-1, but since, so far, we do not interpret x^-2 as (x^-1)^2, it is not a problem.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRDETECT(nlhdlrDetectQuadratic)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlexprdata;
   SCIP_HASHMAP*  expr2idx;
   SCIP_HASHMAP*  seenexpr;
   SCIP_Bool properquadratic;
   int nquadterms = 0;
   int nlinterms = 0;
   int nbilinterms = 0;
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

   /* don't check if enforcement is already ensured */
   if( *enforcedbelow && *enforcedabove )
      return SCIP_OKAY;

   /* if it is not a sum of at least two terms, it cannot be a proper quadratic expressions */
   if( SCIPgetConsExprExprHdlr(expr) != SCIPgetConsExprExprHdlrSum(conshdlr) || SCIPgetConsExprExprNChildren(expr) < 2 )
      return SCIP_OKAY;

#ifdef SCIP_DEBUG
   SCIPinfoMessage(scip, NULL, "Nlhdlr quadratic detecting expr %p aka", (void*)expr);
   SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, expr, NULL) );
   SCIPinfoMessage(scip, NULL, "\n");
   SCIPinfoMessage(scip, NULL, "Have to enforce: Below? %s. Above? %s\n", *enforcedbelow ? "no" : "yes", *enforcedabove ? "no" : "yes");
#endif
   SCIPdebugMsg(scip, "checking if expr %p is a proper quadratic\n", (void*)expr);

   /* check if expression is a proper quadratic expression */
   properquadratic = FALSE;
   SCIP_CALL( SCIPhashmapCreate(&seenexpr, SCIPblkmem(scip), 2*SCIPgetConsExprExprNChildren(expr)) );
   for( c = 0; c < SCIPgetConsExprExprNChildren(expr); ++c )
   {
      SCIP_CONSEXPR_EXPR* child;

      child = SCIPgetConsExprExprChildren(expr)[c];

      assert(child != NULL);

      if( strcmp("pow", SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(child))) == 0 &&
            SCIPgetConsExprExprPowExponent(child) == 2.0 ) /* quadratic term */
      {
         SCIP_CALL( processQuadraticExpr(SCIPgetConsExprExprChildren(child)[0], seenexpr, &properquadratic, &nquadterms,
                  &nlinterms) );
      }
      else if( SCIPgetConsExprExprHdlr(child) == SCIPgetConsExprExprHdlrProduct(conshdlr) &&
            SCIPgetConsExprExprNChildren(child) == 2 ) /* bilinear term */
      {
         ++nbilinterms;
         SCIP_CALL( processQuadraticExpr(SCIPgetConsExprExprChildren(child)[0], seenexpr, &properquadratic, &nquadterms,
                  &nlinterms) );
         SCIP_CALL( processQuadraticExpr(SCIPgetConsExprExprChildren(child)[1], seenexpr, &properquadratic, &nquadterms,
                  &nlinterms) );
      }
      else
      {
         /* first time seen linearly --> assign -1; ++nlinterms
          * not first time --> assign +=1;
          */
         if( SCIPhashmapExists(seenexpr, (void *)child) )
         {
            assert(SCIPhashmapGetImageInt(seenexpr, (void *)child) > 0);

            SCIP_CALL( SCIPhashmapSetImageInt(seenexpr, (void *)child, SCIPhashmapGetImageInt(seenexpr, (void *)child) + 1) );
            properquadratic = TRUE;
         }
         else
         {
            ++nlinterms;
            SCIP_CALL( SCIPhashmapInsertInt(seenexpr, (void *)child, -1) );
         }
      }
   }

   if( ! properquadratic )
   {
      SCIPdebugMsg(scip, "expr %p is not a proper quadratic: can't be handled by us\n", (void*)expr);
      SCIPhashmapFree(&seenexpr);
      return SCIP_OKAY;
   }

   SCIPdebugMsg(scip, "expr %p is proper quadratic: fill data structures\n", (void*)expr);

   /* expr2idx maps expressions to indices; if index > 0, it is its index in the linexprs array, otherwise -index-1 is
    * its index in the quadexprterms array
    */
   SCIP_CALL( SCIPhashmapCreate(&expr2idx, SCIPblkmem(scip), SCIPgetConsExprExprNChildren(expr)) );

   /* allocate memory nlexprdata->nquadexprs, etc */
   SCIP_CALL( SCIPallocClearBlockMemory(scip, nlhdlrexprdata) );
   nlexprdata = *nlhdlrexprdata;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nlexprdata->quadexprterms, nquadterms) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nlexprdata->linexprs, nlinterms) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nlexprdata->lincoefs, nlinterms) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nlexprdata->bilinexprterms, nbilinterms) );

   /* for every term of the expr */
   for( c = 0; c < SCIPgetConsExprExprNChildren(expr); ++c )
   {
      SCIP_CONSEXPR_EXPR* child;
      SCIP_Real coef;

      child = SCIPgetConsExprExprChildren(expr)[c];
      coef = SCIPgetConsExprExprSumCoefs(expr)[c];

      assert(child != NULL);
      assert(coef != 0.0);

      if( strcmp("pow", SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(child))) == 0 &&
            SCIPgetConsExprExprPowExponent(child) == 2.0 ) /* quadratic term */
      {
         SCIP_QUADEXPRTERM* quadexprterm;
         assert(SCIPgetConsExprExprNChildren(child) == 1);

         child = SCIPgetConsExprExprChildren(child)[0];
         assert(SCIPhashmapGetImageInt(seenexpr, (void *)child) > 0);

         /* if expr appeared already, update info, otherwise create info */
         if( SCIPhashmapExists(expr2idx, (void *)child) )
         {
            quadexprterm = &nlexprdata->quadexprterms[SCIPhashmapGetImageInt(expr2idx, (void *)child)];
            assert(quadexprterm->expr == child);

            quadexprterm->sqrcoef = coef;
         }
         else
         {
            SCIP_CALL( SCIPhashmapInsertInt(expr2idx, child, nlexprdata->nquadexprs) );

            quadexprterm = &nlexprdata->quadexprterms[nlexprdata->nquadexprs];
            quadexprterm->expr = child;
            quadexprterm->sqrcoef = coef;
            quadexprterm->lincoef = 0.0;
            quadexprterm->nadjbilin = 0;
            quadexprterm->adjbilinsize = SCIPhashmapGetImageInt(seenexpr, (void *)child);
            SCIP_CALL( SCIPallocBlockMemoryArray(scip, &quadexprterm->adjbilin, quadexprterm->adjbilinsize) );
            ++(nlexprdata->nquadexprs);
         }
      }
      else if( SCIPgetConsExprExprHdlr(child) == SCIPgetConsExprExprHdlrProduct(conshdlr) &&
            SCIPgetConsExprExprNChildren(child) == 2 ) /* bilinear term */
      {
         SCIP_BILINEXPRTERM* bilinexprterm;
         SCIP_CONSEXPR_EXPR* expr1;
         SCIP_CONSEXPR_EXPR* expr2;
         int i;

         assert(SCIPgetConsExprExprProductCoef(child) == 1.0);

         expr1 = SCIPgetConsExprExprChildren(child)[0];
         expr2 = SCIPgetConsExprExprChildren(child)[1];
         assert(expr1 != NULL && expr2 != NULL);

         bilinexprterm = &nlexprdata->bilinexprterms[nlexprdata->nbilinexprterms];

         bilinexprterm->coef = coef;
         if( SCIPhashmapGetImageInt(seenexpr, (void*)expr1) >= SCIPhashmapGetImageInt(seenexpr, (void*)expr2) )
         {
            bilinexprterm->expr1 = expr1;
            bilinexprterm->expr2 = expr2;
         }
         else
         {
            bilinexprterm->expr1 = expr2;
            bilinexprterm->expr2 = expr1;
         }

         for( i = 0; i < 2; ++i )
         {
            SCIP_CONSEXPR_EXPR* bilin;
            SCIP_QUADEXPRTERM* quadexprterm;

            bilin = SCIPgetConsExprExprChildren(child)[i];

            /* if expr appeared already, update info, otherwise create info */
            if( SCIPhashmapExists(expr2idx, (void *)bilin) )
            {
               quadexprterm = &nlexprdata->quadexprterms[SCIPhashmapGetImageInt(expr2idx, (void *)bilin)];
               assert(quadexprterm->expr == bilin);

               quadexprterm->adjbilin[quadexprterm->nadjbilin] = nlexprdata->nbilinexprterms;
               ++(quadexprterm->nadjbilin);
            }
            else
            {
               SCIP_CALL( SCIPhashmapInsertInt(expr2idx, bilin, nlexprdata->nquadexprs) );

               quadexprterm = &nlexprdata->quadexprterms[nlexprdata->nquadexprs];

               quadexprterm->expr = bilin;
               quadexprterm->sqrcoef = 0.0;
               quadexprterm->lincoef = 0.0;
               quadexprterm->nadjbilin = 0;
               quadexprterm->adjbilinsize = SCIPhashmapGetImageInt(seenexpr, (void *)bilin);
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &quadexprterm->adjbilin, quadexprterm->adjbilinsize) );
               quadexprterm->adjbilin[quadexprterm->nadjbilin] = nlexprdata->nbilinexprterms;
               ++(quadexprterm->nadjbilin);
               ++(nlexprdata->nquadexprs);
            }
         }
         nlexprdata->nbilinexprterms++;

         /* TODO: in future store position of second factor in quadexprterms */
         /*bilinexprterm->pos = SCIPhashmapGetImageInt(expr2idx, (void*)bilinexprterm->expr2) */
      }
      else /* linear term */
      {
         if( SCIPhashmapGetImageInt(seenexpr, (void *)child) < 0 )
         {
            assert(SCIPhashmapGetImageInt(seenexpr, (void *)child) == -1);

            /* expression only appears linearly */
            nlexprdata->linexprs[nlexprdata->nlinexprs] = child;
            nlexprdata->lincoefs[nlexprdata->nlinexprs] = coef;
            nlexprdata->nlinexprs++;
         }
         else
         {
            SCIP_QUADEXPRTERM* quadexprterm;
            assert(SCIPhashmapGetImageInt(seenexpr, (void *)child) > 0);

            /* expression will appear non-linearly; if it appeared already, update info, otherwise create info */
            if( SCIPhashmapExists(expr2idx, (void *)child) )
            {
               quadexprterm = &nlexprdata->quadexprterms[SCIPhashmapGetImageInt(expr2idx, (void *)child)];
               assert(quadexprterm->expr == child);

               quadexprterm->lincoef = coef;
            }
            else
            {
               SCIP_CALL( SCIPhashmapInsertInt(expr2idx, child, nlexprdata->nquadexprs) );

               quadexprterm = &nlexprdata->quadexprterms[nlexprdata->nquadexprs];

               quadexprterm->expr = child;
               quadexprterm->sqrcoef = 0.0;
               quadexprterm->lincoef = coef;
               quadexprterm->nadjbilin = 0;
               quadexprterm->adjbilinsize = SCIPhashmapGetImageInt(seenexpr, (void *)child);
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &quadexprterm->adjbilin, quadexprterm->adjbilinsize) );

               ++(nlexprdata->nquadexprs);
            }
         }
      }
   }
   assert(nlexprdata->nquadexprs == nquadterms);
   assert(nlexprdata->nlinexprs == nlinterms);
   assert(nlexprdata->nbilinexprterms == nbilinterms);
   SCIPhashmapFree(&seenexpr);
   SCIPhashmapFree(&expr2idx);

#ifdef DEBUG_DETECT
   /* check structure */
   SCIPinfoMessage(scip, NULL, "Nlhdlr quadratic stored:\n");
   SCIPinfoMessage(scip, NULL, "Linear: \n");
   for( c = 0; c < nlexprdata->nlinexprs; ++c )
   {
      SCIPinfoMessage(scip, NULL, "%g * ", nlexprdata->lincoefs[c]);
      SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, nlexprdata->linexprs[c], NULL) );
      SCIPinfoMessage(scip, NULL, " + ");
   }
   SCIPinfoMessage(scip, NULL, "\n");
   SCIPinfoMessage(scip, NULL, "Quadratic: \n");
   for( c = 0; c < nlexprdata->nquadexprs; ++c )
   {
      SCIPinfoMessage(scip, NULL, "(%g * sqr() + %g) * ", nlexprdata->quadexprterms[c].sqrcoef, nlexprdata->quadexprterms[c].lincoef);
      SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, nlexprdata->quadexprterms[c].expr, NULL) );
      SCIPinfoMessage(scip, NULL, " + ");
   }
   SCIPinfoMessage(scip, NULL, "\n");
   SCIPinfoMessage(scip, NULL, "Bilinear: \n");
   for( c = 0; c < nlexprdata->nbilinexprterms; ++c )
   {
      SCIPinfoMessage(scip, NULL, "%g * ", nlexprdata->bilinexprterms[c].coef);
      SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, nlexprdata->bilinexprterms[c].expr1, NULL) );
      SCIPinfoMessage(scip, NULL, " * ");
      SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, nlexprdata->bilinexprterms[c].expr2, NULL) );
      SCIPinfoMessage(scip, NULL, " + ");
   }
   SCIPinfoMessage(scip, NULL, "\n");
   SCIPinfoMessage(scip, NULL, "Bilinear of quadratics: \n");
   for( c = 0; c < nlexprdata->nquadexprs; ++c )
   {
      int i;
      SCIPinfoMessage(scip, NULL, "For ");
      SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, nlexprdata->quadexprterms[c].expr, NULL) );
      SCIPinfoMessage(scip, NULL, "we see:\n");
      for( i = 0; i < nlexprdata->quadexprterms[c].nadjbilin; ++i )
      {
         SCIPinfoMessage(scip, NULL, "%g * ", nlexprdata->bilinexprterms[nlexprdata->quadexprterms[c].adjbilin[i]].coef);
         SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, nlexprdata->bilinexprterms[nlexprdata->quadexprterms[c].adjbilin[i]].expr1, NULL) );
         SCIPinfoMessage(scip, NULL, " * ");
         SCIP_CALL( SCIPprintConsExprExpr(scip, conshdlr, nlexprdata->bilinexprterms[nlexprdata->quadexprterms[c].adjbilin[i]].expr2, NULL) );
         SCIPinfoMessage(scip, NULL, " + ");
      }
      SCIPinfoMessage(scip, NULL, "\n");
   }
   SCIPinfoMessage(scip, NULL, "\n");
#endif

   /* every detected proper quadratic expression will be handled since we can propagate */
   *success = TRUE;
   *enforcemethods |= SCIP_CONSEXPR_EXPRENFO_INTEVAL | SCIP_CONSEXPR_EXPRENFO_REVERSEPROP;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nlexprdata->quadactivities, nlexprdata->nquadexprs) );

   if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
   {
      /* check if we can do something more: check curvature of quadratic function stored in nlexprdata
       * this is currently only used to decide whether we want to separate, so it can be skipped if in presolve
       */
      SCIPdebugMsg(scip, "expr %p is proper quadratic: checking convexity\n", (void*)expr);
      SCIP_CALL( checkCurvature(scip, nlexprdata) );
   }
   else
   {
      nlexprdata->curvature = SCIP_EXPRCURV_UNKNOWN;
   }

   if( nlexprdata->curvature == SCIP_EXPRCURV_CONVEX )
   {
      SCIPdebugMsg(scip, "expr %p is convex when replacing factors of bilinear terms, bases of squares and every other term by their aux vars\n",
            (void*)expr);

      /* we will estimate the expression from below, that is handle expr <= auxvar */
      *enforcedbelow = TRUE;
      *success = TRUE;
      *enforcemethods |= SCIP_CONSEXPR_EXPRENFO_SEPABELOW;
   }
   else if( nlexprdata->curvature == SCIP_EXPRCURV_CONCAVE )
   {
      SCIPdebugMsg(scip, "expr %p is concave when replacing factors of bilinear terms, bases of squares and every other term by their aux vars\n",
            (void*)expr);

      /* we will estimate the expression from above, that is handle expr >= auxvar */
      *enforcedabove = TRUE;
      *success = TRUE;
      *enforcemethods |= SCIP_CONSEXPR_EXPRENFO_SEPAABOVE;
   }
   else
   {
      /* we cannot do more with this quadratic function */
      return SCIP_OKAY;
   }

   /* quadratic expression is concave/convex -> create aux vars for all expressions stored in nlhdlrexprdata */
   {
      int i;
      SCIP_Bool originalvars = TRUE;

      for( i = 0; i < nlexprdata->nlinexprs; ++i ) /* expressions appearing linearly */
      {
         SCIP_CALL( createAuxVar(scip, conshdlr, nlexprdata->linexprs[i], &originalvars) );
      }
      for( i = 0; i < nlexprdata->nquadexprs; ++i ) /* quadratic terms */
      {
         SCIP_CALL( createAuxVar(scip, conshdlr, nlexprdata->quadexprterms[i].expr, &originalvars) );
      }
      for( i = 0; i < nlexprdata->nbilinexprterms; ++i ) /* bilinear terms */
      {
         SCIP_CALL( createAuxVar(scip, conshdlr, nlexprdata->bilinexprterms[i].expr1, &originalvars) );
         SCIP_CALL( createAuxVar(scip, conshdlr, nlexprdata->bilinexprterms[i].expr2, &originalvars) );
      }

      if( originalvars )
      {
         SCIPsetConsExprExprCurvature(expr, nlexprdata->curvature);
         SCIPdebugMsg(scip, "expr is %s in the original variables\n", nlexprdata->curvature == SCIP_EXPRCURV_CONCAVE ? "concave" : "convex");
      }
   }

   return SCIP_OKAY;
}

/** nonlinear handler auxiliary evaluation callback */
static
SCIP_DECL_CONSEXPR_NLHDLREVALAUX(nlhdlrEvalAuxQuadratic)
{  /*lint --e{715}*/
   int i;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(auxvalue != NULL);

   /* this handler can also handle quadratic expressions whose curvature is unknown or indefinite, since it can
    * propagate them, but it does not separate these
    * we then cannot evaluate w.r.t. auxvars, so we return the value of the expression instead
    */
   if( nlhdlrexprdata->curvature == SCIP_EXPRCURV_UNKNOWN )
   {
      *auxvalue = SCIPgetConsExprExprValue(expr);
      return SCIP_OKAY;
   }

   /* TODO: is this okay or should the constant be stored at the moment of creation? */
   *auxvalue = SCIPgetConsExprExprSumConstant(expr);

   for( i = 0; i < nlhdlrexprdata->nlinexprs; ++i ) /* linear exprs */
      *auxvalue += nlhdlrexprdata->lincoefs[i] * SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(nlhdlrexprdata->linexprs[i]));

   for( i = 0; i < nlhdlrexprdata->nquadexprs; ++i ) /* quadratic terms */
   {
      SCIP_QUADEXPRTERM quadexprterm;
      SCIP_Real solval;

      quadexprterm = nlhdlrexprdata->quadexprterms[i];
      solval = SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(quadexprterm.expr));
      *auxvalue += (quadexprterm.lincoef + quadexprterm.sqrcoef * solval) * solval;
   }

   for( i = 0; i < nlhdlrexprdata->nbilinexprterms; ++i ) /* bilinear terms */
   {
      SCIP_BILINEXPRTERM bilinexprterm;

      bilinexprterm = nlhdlrexprdata->bilinexprterms[i];
      *auxvalue += bilinexprterm.coef *
         SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(bilinexprterm.expr1)) *
         SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(bilinexprterm.expr2));
   }

   return SCIP_OKAY;
}

/** nonlinear handler estimation callback */
static
SCIP_DECL_CONSEXPR_NLHDLRESTIMATE(nlhdlrEstimateQuadratic)
{  /*lint --e{715}*/
   SCIP_Real constant;
   SCIP_Real coef;
   SCIP_Real coef2;
   int j;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(conshdlr != NULL);
   assert(SCIPgetConsExprExprHdlr(expr) == SCIPgetConsExprExprHdlrSum(conshdlr));
   assert(nlhdlrexprdata != NULL);
   assert(rowprep != NULL);
   assert(success != NULL);

   *success = FALSE;

   /* this handler can also handle quadratic expressions whose curvature is unknown or indefinite, since it can
    * propagate them, but it does not separate these
    */
   if( nlhdlrexprdata->curvature == SCIP_EXPRCURV_UNKNOWN )
      return SCIP_OKAY;

   /* if estimating on non-convex side, then do nothing */
   if( ( overestimate && nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONVEX) ||
       (!overestimate && nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONCAVE) )
      return SCIP_OKAY;

   /*
    * compute estimator: quadfun(sol) + \nabla quadfun(sol) (x - sol)
    */

   /* constant */
   SCIPaddRowprepConstant(rowprep, SCIPgetConsExprExprSumConstant(expr));

   /* handle purely linear variables */
   for( j = 0; j < nlhdlrexprdata->nlinexprs; ++j )
   {
      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, SCIPgetConsExprExprAuxVar(nlhdlrexprdata->linexprs[j]),
               nlhdlrexprdata->lincoefs[j]) );
   }

   /* quadratic variables */
   *success = TRUE;
   for( j = 0; j < nlhdlrexprdata->nquadexprs; ++j )
   {
      int k;
      SCIP_VAR* var;
      var = SCIPgetConsExprExprAuxVar(nlhdlrexprdata->quadexprterms[j].expr);

      /* initialize coefficients to linear coefficients of quadratic variables */
      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, var, nlhdlrexprdata->quadexprterms[j].lincoef) );

      /* add linearization of square term */
      coef = 0.0;
      constant = 0.0;
      SCIPaddSquareLinearization(scip, nlhdlrexprdata->quadexprterms[j].sqrcoef, SCIPgetSolVal(scip, sol, var),
         nlhdlrexprdata->quadexprterms[j].nadjbilin == 0 && SCIPvarGetType(var) < SCIP_VARTYPE_CONTINUOUS, &coef, &constant, success);
      if( !*success )
         return SCIP_OKAY;

      SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, var, coef) );
      SCIPaddRowprepConstant(rowprep, constant);

      /* add linearization of bilinear terms that have var as first variable */
      for( k = 0; k < nlhdlrexprdata->quadexprterms[j].nadjbilin; ++k )
      {
         SCIP_BILINEXPRTERM* bilinexprterm;
         SCIP_VAR* var2;

         bilinexprterm = &nlhdlrexprdata->bilinexprterms[nlhdlrexprdata->quadexprterms[j].adjbilin[k]];
         if( SCIPgetConsExprExprAuxVar(bilinexprterm->expr1) != var )
            continue;

         var2 = SCIPgetConsExprExprAuxVar(bilinexprterm->expr2);
         assert(var2 != NULL);
         assert(var2 != var);

         coef = 0.0;
         coef2 = 0.0;
         constant = 0.0;
         SCIPaddBilinLinearization(scip, bilinexprterm->coef, SCIPgetSolVal(scip, sol, var), SCIPgetSolVal(scip, sol,
                  var2), &coef, &coef2, &constant, success);
         if( !*success )
            return SCIP_OKAY;

         SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, var, coef) );
         SCIP_CALL( SCIPaddRowprepTerm(scip, rowprep, var2, coef2) );
         SCIPaddRowprepConstant(rowprep, constant);
      }
   }

   /* merge coefficients that belong to same variable */
   SCIPmergeRowprepTerms(scip, rowprep);

   rowprep->local = FALSE;

   (void) SCIPsnprintf(rowprep->name, SCIP_MAXSTRLEN, "%sestimate_quadratic%p_%s%d",
      overestimate ? "over" : "under",
      (void*)expr,
      sol != NULL ? "sol" : "lp",
      sol != NULL ? SCIPsolGetIndex(sol) : SCIPgetNLPs(scip));

   return SCIP_OKAY;
}

/** nonlinear handler forward propagation callback
 * This method should solve the problem
 * max/min quad expression over box constraints
 * However, this problem is difficult so we are satisfied with a proxy.
 * Interval arithmetic suffices when no variable appears twice, however this is seldom the case, so we try
 * to take care of the dependency problem to some extent:
 * 1. partition the quadratic expression as sum of quadratic functions
 * \sum_l q_l
 * where q_l = a_l expr_l^2 + \sum_{i \in P_l} b_il expr_i expr_l + c_l expr_l
 * 2. build interval quadratic functions, i.e, a x^2 + b x where b is an interval as
 * a_l expr_l^2 + [\sum_{i \in P_l} b_il expr_i + c_l] expr_l
 * 3. compute \min and \max { a x^2 + b x : x \in [x] } for each interval quadratic, i.e.
 * \min and \max a_l expr_l^2 + [\sum_{i \in P_l} b_il expr_i + c_l] expr_l : expr_l \in [expr_l]
 *
 * In particular, P_l = \{i : expr_l expr_i is a bilinear expr\}. Note that the
 * order matters, that is in P_l, expr_l is the the first expression.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRINTEVAL(nlhdlrIntevalQuadratic)
{ /*lint --e{715}*/

   SCIP_INTERVAL quadactivity;

   assert(scip != NULL);
   assert(expr != NULL);

   assert(nlhdlrexprdata != NULL);
   assert(nlhdlrexprdata->nquadexprs != 0);

   SCIPdebugMsg(scip, "Interval evaluation of quadratic expr\n");

   /*
    * compute activity of linear part
    */
   {
      int i;

      SCIPdebugMsg(scip, "Computing activity of linear part\n");

      SCIPintervalSet(&nlhdlrexprdata->linactivity, SCIPgetConsExprExprSumConstant(expr));
      for( i = 0; i < nlhdlrexprdata->nlinexprs; ++i )
      {
         SCIP_INTERVAL linterminterval;

         SCIPintervalMulScalar(SCIP_INTERVAL_INFINITY, &linterminterval,
               SCIPgetConsExprExprInterval(nlhdlrexprdata->linexprs[i]), nlhdlrexprdata->lincoefs[i]);
         SCIPintervalAdd(SCIP_INTERVAL_INFINITY, &nlhdlrexprdata->linactivity, nlhdlrexprdata->linactivity, linterminterval);
      }

      SCIPdebugMsg(scip, "Activity of linear part is [%g, %g]\n", nlhdlrexprdata->linactivity.inf,
            nlhdlrexprdata->linactivity.sup);
   }

   /*
    * compute activity of quadratic part
    */
   nlhdlrexprdata->nneginfinityquadact = 0;
   nlhdlrexprdata->nposinfinityquadact = 0;
   nlhdlrexprdata->minquadfiniteact = 0.0;
   nlhdlrexprdata->maxquadfiniteact = 0.0;
   SCIPintervalSet(&quadactivity, 0.0);
   {
      SCIP_BILINEXPRTERM* bilinterms;
      int i;

      SCIPdebugMsg(scip, "Computing activity of quadratic part\n");

      bilinterms = nlhdlrexprdata->bilinexprterms;
      for( i = 0; i < nlhdlrexprdata->nquadexprs; ++i )
      {
         int j;
         SCIP_INTERVAL b;
         SCIP_QUADEXPRTERM* quadexpr;
         SCIP_Real quadub;
         SCIP_Real quadlb;

         /* b = [c_l] */
         quadexpr = &nlhdlrexprdata->quadexprterms[i];
         SCIPintervalSet(&b, quadexpr->lincoef);
         for( j = 0; j < quadexpr->nadjbilin; ++j )
         {
            SCIP_BILINEXPRTERM* bilinterm;
            SCIP_INTERVAL bterm;

            bilinterm = &bilinterms[quadexpr->adjbilin[j]];
            if( bilinterm->expr1 != quadexpr->expr )
               continue;

            /* b += [b_jl * expr_j] for j \in P_l */
            SCIPintervalMulScalar(SCIP_INTERVAL_INFINITY, &bterm, SCIPgetConsExprExprInterval(bilinterm->expr2),
                  bilinterm->coef);
            SCIPintervalAdd(SCIP_INTERVAL_INFINITY, &b, b, bterm);

#ifdef DEBUG_PROP
         SCIPinfoMessage(scip, NULL, "b += %g * [expr2], where <expr2> is:", bilinterm->coef);
         SCIP_CALL( SCIPprintConsExprExpr(scip, SCIPfindConshdlr(scip, "expr"), bilinterm->expr2, NULL) );
         SCIPinfoMessage(scip, NULL, "\n");
#endif
         }

         /* TODO: under which assumptions do we know that we just need to compute min or max? its probably the locks that give some information here */
         quadub = SCIPintervalQuadUpperBound(SCIP_INTERVAL_INFINITY, quadexpr->sqrcoef, b,
               SCIPgetConsExprExprInterval(quadexpr->expr));

         /* TODO: implement SCIPintervalQuadLowerBound */
         {
            SCIP_INTERVAL minusb;
            SCIPintervalSetBounds(&minusb, -SCIPintervalGetSup(b), -SCIPintervalGetInf(b));

            quadlb = -SCIPintervalQuadUpperBound(SCIP_INTERVAL_INFINITY, -quadexpr->sqrcoef, minusb,
                  SCIPgetConsExprExprInterval(quadexpr->expr));
         }

#ifdef DEBUG_PROP
         SCIPinfoMessage(scip, NULL, "Computing activity for quadratic term a <expr>^2 + b <expr>, where <expr> is:");
         SCIP_CALL( SCIPprintConsExprExpr(scip, SCIPfindConshdlr(scip, "expr"), quadexpr->expr, NULL) );
         SCIPinfoMessage(scip, NULL, "\n");
         SCIPinfoMessage(scip, NULL, "a = %g, b = [%g, %g] and activity [%g, %g]\n", quadexpr->sqrcoef, b.inf, b.sup, quadlb, quadub);
#endif

         SCIPintervalSetBounds(&nlhdlrexprdata->quadactivities[i], quadlb, quadub);
         SCIPintervalAdd(SCIP_INTERVAL_INFINITY, &quadactivity, quadactivity, nlhdlrexprdata->quadactivities[i]);

         /* get number of +/-infinity contributions and compute finite activity */
         if( quadlb <= -SCIP_INTERVAL_INFINITY )
            nlhdlrexprdata->nneginfinityquadact++;
         else
         {
            SCIP_ROUNDMODE roundmode;

            roundmode = SCIPintervalGetRoundingMode();
            SCIPintervalSetRoundingModeDownwards();

            nlhdlrexprdata->minquadfiniteact += quadlb;

            SCIPintervalSetRoundingMode(roundmode);
         }
         if( quadub >= SCIP_INTERVAL_INFINITY )
            nlhdlrexprdata->nposinfinityquadact++;
         else
         {
            SCIP_ROUNDMODE roundmode;

            roundmode = SCIPintervalGetRoundingMode();
            SCIPintervalSetRoundingModeUpwards();

            nlhdlrexprdata->maxquadfiniteact += quadub;

            SCIPintervalSetRoundingMode(roundmode);
         }
      }

      SCIPdebugMsg(scip, "Activity of quadratic part is [%g, %g]\n", quadactivity.inf, quadactivity.sup);
   }

   /* interval evaluation is linear activity + quadactivity */
   SCIPintervalAdd(SCIP_INTERVAL_INFINITY, interval, nlhdlrexprdata->linactivity,  quadactivity);

   return SCIP_OKAY;
}


/** nonlinear handler reverse propagation callback
 * @note: the implemented technique is a proxy for solving the OBBT problem min/max{ x_i : quad expr \in [quad expr] }
 * and as such can be improved.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - nlhdlr : nonlinear handler
 *  - expr : expression
 *  - nlhdlrexprdata : expression specific data of the nonlinear handler
 *  - reversepropqueue : expression queue in reverse propagation, to be passed on to SCIPtightenConsExprExprInterval
 *  - infeasible: buffer to store whether an expression's bounds were propagated to an empty interval
 *  - nreductions : buffer to store the number of interval reductions of all children
 *  - force : force tightening even if it is below the bound strengthening tolerance
 */
static
SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP(nlhdlrReversepropQuadratic)
{ /*lint --e{715}*/
   SCIP_INTERVAL rhs;
   SCIP_INTERVAL quadactivity;

   SCIPdebugMsg(scip, "Reverse propagation of quadratic expr\n");

   assert(scip != NULL);
   assert(expr != NULL);
   assert(reversepropqueue != NULL);
   assert(infeasible != NULL);

   /* not possible to conclude finite bounds if the interval of the expression is [-inf,inf] */
   if( SCIPintervalIsEntire(SCIP_INTERVAL_INFINITY, SCIPgetConsExprExprInterval(expr)) )
      return SCIP_OKAY;

   /* propagate linear part in rhs = expr's interval - quadratic activity; first, reconstruct the quadratic activity */
   SCIPintervalSetBounds(&quadactivity,
         nlhdlrexprdata->nneginfinityquadact > 0 ? -SCIP_INTERVAL_INFINITY : nlhdlrexprdata->minquadfiniteact,
         nlhdlrexprdata->nposinfinityquadact > 0 ?  SCIP_INTERVAL_INFINITY : nlhdlrexprdata->maxquadfiniteact);

   SCIPintervalSub(SCIP_INTERVAL_INFINITY, &rhs, SCIPgetConsExprExprInterval(expr), quadactivity);
   SCIP_CALL( SCIPreverseConsExprExprPropagateWeightedSum(scip, nlhdlrexprdata->nlinexprs,
            nlhdlrexprdata->linexprs, nlhdlrexprdata->lincoefs, SCIPgetConsExprExprSumConstant(expr),
            rhs, reversepropqueue, infeasible, nreductions, force) );

   /* stop if we find infeasibility */
   if( *infeasible )
      return SCIP_OKAY;

   /* propagate quadratic part in expr's interval - linear activity:
    * linear activity was computed in INTEVAL
    * One way of achieving this is by, for each expression expr_i, write the quadratic expression as
    * a_i expr^2_i + expr_i ( \sum_{j \in J_i} b_ij expr_j + c_i ) + quadratic expression in expr_k for k \neq i
    * then compute the interval b = [\sum_{j \in J_i} b_ij expr_j + c_i], where J_i are all the indices j such that the
    * bilinear expression expr_i expr_j appears, and use some technique (like the one in nlhdlrIntervalQuadratic), to
    * evaluate the activity rest_i = [quadratic expression in expr_k for k \neq i].
    * Then, solve a_i expr_i^2 + b expr_i = [expr] - rest_i =: rhs_i.
    * However, this might be expensive, specially computing rest_i. Hence, we implement a simpler version, namely,
    * we use the same partition as in nlhdlrIntervalQuadratic for the bilinear terms. This way,
    * b = [\sum_{j \in P_i} b_ij expr_j + c_i], where P_i is defined as in nlhdlrIntervalQuadratic, all the indices j
    * such that expr_i expr_j appears in that order, and rest_i = sum_{k \neq i} [\min q_k, \max q_k] where
    * q_k = a_k expr_k^2 + [\sum_{j \in P_k} b_jk expr_j + c_k] expr_k. The intervals [\min q_k, \max q_k] were
    * already computed in nlhdlrIntervalQuadratic, so we just reuse them.
    *
    * TODO: in cons_quadratic there seems to be a further technique that tries, when propagating expr_i, to borrow a
    * bilinear term expr_k expr_i when the quadratic function for expr_k is simple enough.
    *
    * TODO: handle simple cases
    * TODO: identify early when there is nothing to be gain
    */
   SCIPintervalSub(SCIP_INTERVAL_INFINITY, &rhs, SCIPgetConsExprExprInterval(expr), nlhdlrexprdata->linactivity);
   {
      SCIP_BILINEXPRTERM* bilinterms;
      int i;

      bilinterms = nlhdlrexprdata->bilinexprterms;
      for( i = 0; i < nlhdlrexprdata->nquadexprs; ++i )
      {
         int j;
         SCIP_INTERVAL b;
         SCIP_INTERVAL rhs_i;
         SCIP_INTERVAL rest_i;
         SCIP_QUADEXPRTERM quadexpr;

         /* b = [c_l] */
         quadexpr = nlhdlrexprdata->quadexprterms[i];
         SCIPintervalSet(&b, quadexpr.lincoef);
         for( j = 0; j < quadexpr.nadjbilin; ++j )
         {
            SCIP_BILINEXPRTERM bilinterm;
            SCIP_INTERVAL bterm;

            bilinterm = bilinterms[quadexpr.adjbilin[j]];
            if( bilinterm.expr1 != quadexpr.expr )
               continue;

            /* b += [b_jl * expr_j] for j \in P_l */
            SCIPintervalMulScalar(SCIP_INTERVAL_INFINITY, &bterm, SCIPgetConsExprExprInterval(bilinterm.expr2),
                  bilinterm.coef);
            SCIPintervalAdd(SCIP_INTERVAL_INFINITY, &b, b, bterm);
         }

         /* rhs_i = rhs - rest_i.
          * to compute rest_i = [\sum_{k \neq i} q_k] we just have to substract
          * the activity of q_i from quadactivity; however, care must be taken about infinities;
          * if [q_i].sup = +infinity and there is = 1 contributing +infinity -> rest_i.sup = maxquadfiniteact
          * if [q_i].sup = +infinity and there is > 1 contributing +infinity -> rest_i.sup = +infinity
          * if [q_i].sup = finite and there is > 0 contributing +infinity -> rest_i.sup = +infinity
          * if [q_i].sup = finite and there is = 0 contributing +infinity -> rest_i.sup = maxquadfiniteact - [q_i].sup
          *
          * the same holds when replacing sup with inf, + with - and max(quadfiniteact) with min(...)
          */
         /* compute rest_i.sup */
         if( SCIPintervalGetSup(nlhdlrexprdata->quadactivities[i]) < SCIP_INTERVAL_INFINITY &&
               nlhdlrexprdata->nposinfinityquadact == 0 )
         {
            SCIP_ROUNDMODE roundmode;

            roundmode = SCIPintervalGetRoundingMode();
            SCIPintervalSetRoundingModeUpwards();
            rest_i.sup = nlhdlrexprdata->maxquadfiniteact - SCIPintervalGetSup(nlhdlrexprdata->quadactivities[i]);

            SCIPintervalSetRoundingMode(roundmode);
         }
         else if( SCIPintervalGetSup(nlhdlrexprdata->quadactivities[i]) >= SCIP_INTERVAL_INFINITY &&
               nlhdlrexprdata->nposinfinityquadact == 1 )
            rest_i.sup = nlhdlrexprdata->maxquadfiniteact;
         else
            rest_i.sup = SCIP_INTERVAL_INFINITY;

         /* compute rest_i.inf */
         if( SCIPintervalGetInf(nlhdlrexprdata->quadactivities[i]) > -SCIP_INTERVAL_INFINITY &&
               nlhdlrexprdata->nneginfinityquadact == 0 )
         {
            SCIP_ROUNDMODE roundmode;

            roundmode = SCIPintervalGetRoundingMode();
            SCIPintervalSetRoundingModeDownwards();
            rest_i.inf = nlhdlrexprdata->minquadfiniteact - SCIPintervalGetInf(nlhdlrexprdata->quadactivities[i]);

            SCIPintervalSetRoundingMode(roundmode);
         }
         else if( SCIPintervalGetInf(nlhdlrexprdata->quadactivities[i]) <= -SCIP_INTERVAL_INFINITY &&
               nlhdlrexprdata->nneginfinityquadact == 1 )
            rest_i.inf = nlhdlrexprdata->minquadfiniteact;
         else
            rest_i.inf = -SCIP_INTERVAL_INFINITY;

#if 0 /* I (SV) added the following in cons_quadratic to fix/workaround some bug. Maybe we'll need this here, too? */
         /* FIXME in theory, rest_i should not be empty here
          * what we tried to do here is to remove the contribution of the i'th bilinear term (=bilinterm) to [minquadactivity,maxquadactivity] from rhs
          * however, quadactivity is computed differently (as x*(a1*y1+...+an*yn)) than q_i (a*ak*yk) and since interval arithmetics do overestimation,
          * it can happen that q_i is actually slightly larger than quadactivity, which results in rest_i being (slightly) empty
          * a proper fix could be to compute the quadactivity also as x*a1*y1+...+x*an*yn if sqrcoef=0, but due to taking
          * also infinite bounds into account, this complicates the code even further
          * instead, I'll just work around this by turning an empty rest_i into a small non-empty one
          */
         if( SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, rest_i) )
         {
            assert(SCIPisSumRelEQ(scip, rest_i.inf, rest_i.sup));
            SCIPswapReals(&rest_i.inf, &rest_i.sup);
         }
#endif
         assert(!SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, rest_i));

         /* compute rhs_i */
         SCIPintervalSub(SCIP_INTERVAL_INFINITY, &rhs_i, rhs, rest_i);

         /* solve a_i expr_i^2 + b expr_i = rhs_i */
         if( SCIPintervalIsEntire(SCIP_INTERVAL_INFINITY, rhs_i) )
            continue;

         SCIP_CALL( propagateBoundsQuadExpr(scip, quadexpr, b, rhs_i, reversepropqueue, infeasible, nreductions, force) );

         /* stop if we find infeasibility */
         if( *infeasible )
            return SCIP_OKAY;
      }
   }

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE(nlhdlrBranchscoreQuadratic)
{ /*lint --e{715}*/
   SCIP_Real side;
   SCIP_Real violation;
   int i;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(nlhdlrexprdata != NULL);
   assert(success != NULL);

   *success = FALSE;

   /* this handler can also handle quadratic expressions whose curvature is unknown or indefinite, since it can
    * propagate them; however, we only separate for convex quadratics, so we only provide branchscore in that case
    * normally, we should not need to branch, but there could be small violations or numerical issues that
    * prevented separation to succeed
    */
   if( nlhdlrexprdata->curvature == SCIP_EXPRCURV_UNKNOWN )
      return SCIP_OKAY;

   assert(nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONVEX || nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONCAVE);

   side = SCIPgetSolVal(scip, sol, SCIPgetConsExprExprAuxVar(expr));

   SCIPdebugMsg(scip, "Activity = %g (act of expr is %g), side = %g, curvature %s\n", auxvalue,
      SCIPgetConsExprExprValue(expr), side, nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONVEX ? "convex" :
         "concave");

   /* if convex, then we enforce expr <= auxvar, so violation is expr - auxvar = activity - side, if positive
    * if concave, then we enforce expr >= auxvar, so violation is auxvar - expr = side - activity, if positive
    */
   if( nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONVEX )
      violation = MAX(0.0, auxvalue - side);
   else /* nlhdlrexprdata->curvature == SCIP_EXPRCURV_CONCAVE */
      violation = MAX(0.0, side - auxvalue);

   /* if there is violation, then add branchscore for all expr in quadratic part */
   if( violation > 0.0 )
   {
      for( i = 0; i < nlhdlrexprdata->nquadexprs; ++i )
         SCIPaddConsExprExprBranchScore(scip, nlhdlrexprdata->quadexprterms[i].expr, brscoretag, violation);

      *success = TRUE;
   }

   return SCIP_OKAY;
}

/** nonlinear handler copy callback
 *
 * the method includes the nonlinear handler into a expression constraint handler
 *
 * This method is usually called when doing a copy of an expression constraint handler.
 */
static
SCIP_DECL_CONSEXPR_NLHDLRCOPYHDLR(nlhdlrcopyHdlrQuadratic)
{  /*lint --e{715}*/
   assert(targetscip != NULL);
   assert(targetconsexprhdlr != NULL);
   assert(sourcenlhdlr != NULL);
   assert(strcmp(SCIPgetConsExprNlhdlrName(sourcenlhdlr), NLHDLR_NAME) == 0);

   SCIP_CALL( SCIPincludeConsExprNlhdlrQuadratic(targetscip, targetconsexprhdlr) );

   return SCIP_OKAY;
}

/** includes quadratic nonlinear handler to consexpr */
SCIP_RETCODE SCIPincludeConsExprNlhdlrQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_NLHDLR* nlhdlr;

   assert(scip != NULL);
   assert(consexprhdlr != NULL);

   SCIP_CALL( SCIPincludeConsExprNlhdlrBasic(scip, consexprhdlr, &nlhdlr, NLHDLR_NAME, NLHDLR_DESC, NLHDLR_PRIORITY,
            nlhdlrDetectQuadratic, nlhdlrEvalAuxQuadratic, NULL) );

   SCIPsetConsExprNlhdlrCopyHdlr(scip, nlhdlr, nlhdlrcopyHdlrQuadratic);
   SCIPsetConsExprNlhdlrFreeExprData(scip, nlhdlr, nlhdlrfreeExprDataQuadratic);
   SCIPsetConsExprNlhdlrSepa(scip, nlhdlr, NULL, NULL, nlhdlrEstimateQuadratic, NULL);
   SCIPsetConsExprNlhdlrProp(scip, nlhdlr, nlhdlrIntevalQuadratic, nlhdlrReversepropQuadratic);
   SCIPsetConsExprNlhdlrBranchscore(scip, nlhdlr, nlhdlrBranchscoreQuadratic);

   return SCIP_OKAY;
}
