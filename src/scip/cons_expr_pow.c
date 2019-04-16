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

/**@file   cons_expr_pow.c
 * @brief  power expression handler
 * @author Benjamin Mueller
 *
 * @todo initsepaPow
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/*lint --e{835}*/

#include <string.h>

#include "scip/cons_expr_value.h"
#include "scip/cons_expr_var.h"
#include "scip/cons_expr_pow.h"
#include "scip/cons_expr_product.h"
#include "scip/cons_expr_sum.h"

#define EXPRHDLR_NAME         "pow"
#define EXPRHDLR_DESC         "power expression"
#define EXPRHDLR_PRECEDENCE  55000
#define EXPRHDLR_HASHKEY     SCIPcalcFibHash(21163.0)

/*
 * Data structures
 */

#define SIGNPOW_ROOTS_KNOWN 10                  /**< up to which (integer) exponents precomputed roots have been stored */

/** The positive root of the polynomial (n-1) y^n + n y^(n-1) - 1 is needed in separation.
 *  Here we store these roots for small integer values of n.
 */
static
SCIP_Real signpow_roots[SIGNPOW_ROOTS_KNOWN+1] = {
   -1.0,                     /* no root for n=0 */
   -1.0,                     /* no root for n=1 */
   0.41421356237309504880,   /* root for n=2 (-1+sqrt(2)) */
   0.5,                      /* root for n=3 */
   0.56042566045031785945,   /* root for n=4 */
   0.60582958618826802099,   /* root for n=5 */
   0.64146546982884663257,   /* root for n=6 */
   0.67033204760309682774,   /* root for n=7 */
   0.69428385661425826738,   /* root for n=8 */
   0.71453772716733489700,   /* root for n=9 */
   0.73192937842370733350    /* root for n=10 */
};


struct SCIP_ConsExpr_ExprData
{
   SCIP_Real  exponent;     /**< exponent */
   SCIP_Real  root;         /**< root for signpower, or SCIP_INVALID if not signpower */
};

/*
 * Local methods
 */

/** computes positive root of the polynomial (n-1) y^n + n y^(n-1) - 1 for n > 1 */
static
SCIP_RETCODE computeSignpowerRoot(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            root,               /**< buffer where to store computed root */
   SCIP_Real             exponent            /**< exponent n */
   )
{
   SCIP_Real polyval;
   SCIP_Real gradval;
   int iter;

   assert(scip != NULL);
   assert(exponent > 1.0);
   assert(root != NULL);

   /* lookup for popular integer exponent */
   if( SCIPisIntegral(scip, exponent) && exponent-0.5 < SIGNPOW_ROOTS_KNOWN )
   {
      *root = signpow_roots[(int)SCIPfloor(scip, exponent+0.5)];
      return SCIP_OKAY;
   }

   /* loopup for weymouth exponent */
   if( SCIPisEQ(scip, exponent, 1.852) )
   {
      *root = 0.39821689389382575186;
      return SCIP_OKAY;
   }

   /* search for a positive root of (n-1) y^n + n y^(n-1) - 1
    * use the closest precomputed root as starting value
    */
   if( exponent >= SIGNPOW_ROOTS_KNOWN )
      *root = signpow_roots[SIGNPOW_ROOTS_KNOWN];
   else if( exponent <= 2.0 )
      *root = signpow_roots[2];
   else
      *root = signpow_roots[(int)SCIPfloor(scip, exponent)];

   for(iter = 0; iter < 1000; ++iter )
   {
      polyval = (exponent - 1.0) * pow(*root, exponent) + exponent * pow(*root, exponent - 1.0) - 1.0;
      if( fabs(polyval) < 1e-12 && SCIPisZero(scip, polyval) )
         break;

      /* gradient of (n-1) y^n + n y^(n-1) - 1 is n(n-1)y^(n-1) + n(n-1)y^(n-2) */
      gradval = (exponent - 1.0) * exponent * (pow(*root, exponent - 1.0) + pow(*root, exponent - 2.0));
      if( SCIPisZero(scip, gradval) )
         break;

      /* update root by adding -polyval/gradval (Newton's method) */
      *root -= polyval / gradval;
      if( *root < 0.0 )
         *root = 0.0;
   }

   if( !SCIPisZero(scip, polyval) )
   {
      SCIPerrorMessage("failed to compute root for exponent %g\n", exponent);
      return SCIP_ERROR;
   }
   SCIPdebugMsg(scip, "root for %g is %.20g, certainty = %g\n", exponent, *root, polyval);
   /* @todo cache root value for other expressions (an exponent seldom comes alone)?? (they are actually really fast to compute...) */

   return SCIP_OKAY;
}

/** computes negative root of the polynomial (n-1) y^n - n y^(n-1) + 1 for n < -1 */
static
SCIP_RETCODE computeHyperbolaRoot(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            root,               /**< buffer where to store computed root */
   SCIP_Real             exponent            /**< exponent n */
   )
{
   SCIP_Real polyval;
   SCIP_Real gradval;
   int iter;

   assert(scip != NULL);
   assert(exponent < -1.0);
   assert(root != NULL);

   *root = -2.0;  /* that's the solution for n=-2 */

   for(iter = 0; iter < 1000; ++iter )
   {
      polyval = (exponent - 1.0) * pow(*root, exponent) - exponent * pow(*root, exponent - 1.0) + 1.0;
      if( fabs(polyval) < 1e-12 && SCIPisZero(scip, polyval) )
         break;

      /* gradient of (n-1) y^n - n y^(n-1) + 1 is n(n-1)y^(n-1) - n(n-1)y^(n-2) */
      gradval = (exponent - 1.0) * exponent * (pow(*root, exponent - 1.0) - pow(*root, exponent - 2.0));
      if( SCIPisZero(scip, gradval) )
         break;

      /* update root by adding -polyval/gradval (Newton's method) */
      *root -= polyval / gradval;
      if( *root >= 0.0 )
         *root = -1;
   }

   if( !SCIPisZero(scip, polyval) )
   {
      SCIPerrorMessage("failed to compute root for exponent %g\n", exponent);
      return SCIP_ERROR;
   }
   SCIPdebugMsg(scip, "root for %g is %.20g, certainty = %g\n", exponent, *root, polyval);
   /* @todo cache root value for other expressions (an exponent seldom comes alone)?? (they are actually really fast to compute...) */

   return SCIP_OKAY;
}


static
SCIP_RETCODE createData(
   SCIP*                    scip,            /**< SCIP data structure */
   SCIP_CONSEXPR_EXPRDATA** exprdata,        /**< pointer where to store expression data */
   SCIP_Real                exponent         /**< exponent of the power expression */
   )
{
   assert(exprdata != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, exprdata) );

   (*exprdata)->exponent = exponent;
   (*exprdata)->root = SCIP_INVALID;

   return SCIP_OKAY;
}

/** computes a tangent at a reference point by linearization
 *
 * linearization in xref is xref^exponent + exponent * xref^(exponent-1) (x - xref)
 * = (1-exponent) * xref^exponent + exponent * xref^(exponent-1) * x
 */
static
void computeTangent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             exponent,           /**< exponent */
   SCIP_Real             xref,               /**< reference point where to linearize */
   SCIP_Real*            constant,           /**< buffer to store constant term of secant */
   SCIP_Real*            slope,              /**< buffer to store slope of secant */
   SCIP_Bool*            success             /**< buffer to store whether secant could be computed */
)
{
   SCIP_Real xrefpow;

   assert(scip != NULL);
   assert(constant != NULL);
   assert(slope != NULL);
   assert(success != NULL);
   assert(xref != 0.0 || exponent > 0.0);
   assert(EPSISINT(exponent, 0.0) || !SCIPisNegative(scip, xref)); /* non-integral exponent -> reference point must be >= 0 */

   /* TODO power is not differentiable at 0.0 for exponent < 0
    * should we forbid here that xref > 0, do something smart here, or just return success=FALSE?
    */
   /* assert(exponent >= 1.0 || xref > 0.0); */

   if( !EPSISINT(exponent, 0.0) && xref < 0.0 )
      xref = 0.0;

   xrefpow = pow(xref, exponent - 1.0);

   /* if huge xref and/or exponent too large, then pow may overflow */
   if( !SCIPisFinite(xrefpow) )
   {
      *success = FALSE;
      return;
   }

   *constant = (1.0 - exponent) * xrefpow * xref;
   *slope = exponent * xrefpow;
   *success = TRUE;
}

/** computes a secant between lower and upper bound
 *
 * secant is xlb^exponent + (xub^exponent - xlb^exponent) / (xub - xlb) * (x - xlb)
 * = xlb^exponent - slope * xlb + slope * x  with slope = (xub^exponent - xlb^exponent) / (xub - xlb)
 */
static
void computeSecant(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             exponent,           /**< exponent */
   SCIP_Real             xlb,                /**< lower bound on x */
   SCIP_Real             xub,                /**< upper bound on x */
   SCIP_Real*            constant,           /**< buffer to store constant term of secant */
   SCIP_Real*            slope,              /**< buffer to store slope of secant */
   SCIP_Bool*            success             /**< buffer to store whether secant could be computed */
)
{
   assert(scip != NULL);
   assert(constant != NULL);
   assert(slope != NULL);
   assert(success != NULL);
   assert(!SCIPisEQ(scip, xlb, xub)); /* taken care of in separatePointPow */
   assert(xlb >= 0.0 || EPSISINT(exponent, 0.0));
   assert(xub >= 0.0 || EPSISINT(exponent, 0.0));
   assert(exponent != 1.0);

   *success = FALSE;

   /* infinite bounds will not work */
   if( SCIPisInfinity(scip, -xlb) || SCIPisInfinity(scip, xub) )
      return;

   /* first handle some special cases */
   if( EPSISINT(exponent / 2.0, 0.0) && xub > 0.1 && SCIPisFeasEQ(scip, xlb, -xub) )
   {
      /* for even exponents with xlb ~ -xub the slope would be very close to 0
       * since xub^n - xlb^n is prone to cancellation here, we omit computing this secant (it's probably useless)
       * unless the bounds are close to 0 as well (xub <= 0.1 in the "if" above)
       * or we have exactly xlb=-xub, where we can return a clean 0.0 (though it's probably useless)
       */
      if( xlb == -xub ) /*lint !e777*/
      {
         *slope = 0.0;
         *constant = pow(xlb, exponent);
      }
      else
      {
         assert(SCIPisFeasZero(scip, (pow(xub,exponent)-pow(xlb,exponent))/(xub-xlb)));
         return;
      }
   }
   else if( xlb == 0.0 && exponent > 0.0 ) /*lint !e777*/
   {
      *slope = pow(xub, exponent-1.0);
      *constant = 0.0;
   }
   else if( xub == 0.0 && exponent > 0.0 ) /*lint !e777*/
   {
      *slope = pow(xlb, exponent-1.0);
      *constant = 0.0;
   }
   else
   {
      SCIP_Real lbval;
      SCIP_Real ubval;

      lbval = pow(xlb, exponent);
      if( !SCIPisFinite(lbval) )
         return;

      ubval = pow(xub, exponent);
      if( !SCIPisFinite(ubval) )
         return;

      /* this can have bad numerics when xlb^exponent and xub^exponent are very close
       * for now, only check that things did not cancel out completely
       * - the secant would be ok, if SCIPisEQ(xlb, xub), but this is already excluded above
       * - the secant would be ok, if SCIPisEQ(xlb, -xub) and the exponent is even, but this is already handled above
       */
      if( lbval == ubval ) /*lint !e777*/
         return;

      *slope = (ubval - lbval) / (xub - xlb);
      *constant = lbval - *slope * xlb;
   }

   /* check whether we had overflows */
   if( !SCIPisFinite(*slope) || !SCIPisFinite(*constant) )
      return;

   *success = TRUE;
}

/** Separation for parabola
 *
 * - even positive powers: x^2, x^4, x^6 with x arbitrary, or
 * - positive powers > 1: x^1.5, x^2.5 with x >= 0

  100 +--------------------------------------------------------------------+
      |*               +                 +                +               *|
   90 |**                                                     x**2 ********|
      |  *                                                              *  |
   80 |-+*                                                              *+-|
      |   **                                                          **   |
   70 |-+   *                                                        *   +-|
      |     **                                                      **     |
   60 |-+     *                                                    *     +-|
      |       **                                                  **       |
   50 |-+       *                                                *       +-|
      |          **                                            **          |
   40 |-+          *                                          *          +-|
      |             **                                      **             |
   30 |-+            **                                    **            +-|
      |                **                                **                |
   20 |-+                **                            **                +-|
      |                   ***                        ***                   |
   10 |-+                   ***                    ***                   +-|
      |                +       *****     +    *****       +                |
    0 +--------------------------------------------------------------------+
     -10              -5                 0                5                10

 */
static
void estimateParabola(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             exponent,           /**< exponent */
   SCIP_Bool             overestimate,       /**< should the power be overestimated? */
   SCIP_Real             xlb,                /**< lower bound on x */
   SCIP_Real             xub,                /**< upper bound on x */
   SCIP_Real             xref,               /**< reference point (where to linearize) */
   SCIP_Real*            constant,           /**< buffer to store constant term of estimator */
   SCIP_Real*            slope,              /**< buffer to store slope of estimator */
   SCIP_Bool*            islocal,            /**< buffer to store whether estimator only locally valid, that is, depends on given bounds */
   SCIP_Bool*            success             /**< buffer to store whether estimator could be computed */
)
{
   assert(scip != NULL);
   assert(constant != NULL);
   assert(slope != NULL);
   assert(islocal != NULL);
   assert(success != NULL);
   assert((exponent >= 0.0 && EPSISINT(exponent/2.0, 0.0)) || (exponent > 1.0 && xlb >= 0.0));

   if( !overestimate )
   {
      computeTangent(scip, exponent, xref, constant, slope, success);
      *islocal = FALSE;
   }
   else
   {
      /* overestimation -> secant */
      computeSecant(scip, exponent, xlb, xub, constant, slope, success);
      *islocal = TRUE;
   }
}


/** Separation for signpower
 *
 * - odd positive powers, x^3, x^5, x^7
 * - sign(x)|x|^n for n > 1
 * - lower bound on x is negative (otherwise one should use separation for parabola)

  100 +--------------------------------------------------------------------+
      |                +                 +                +              **|
      |                                                   x*abs(x) ******* |
      |                                                              **    |
      |                                                            **      |
   50 |-+                                                       ***      +-|
      |                                                      ***           |
      |                                                   ***              |
      |                                               *****                |
      |                                          *****                     |
    0 |-+                        ****************                        +-|
      |                     *****                                          |
      |                *****                                               |
      |              ***                                                   |
      |           ***                                                      |
  -50 |-+      ***                                                       +-|
      |      **                                                            |
      |    **                                                              |
      |  **                                                                |
      |**              +                 +                +                |
 -100 +--------------------------------------------------------------------+
     -10              -5                 0                5                10

 */
static
void estimateSignpower(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             exponent,           /**< exponent */
   SCIP_Real             root,               /**< positive root of the polynomial (n-1) y^n + n y^(n-1) - 1, if xubglobal > 0 */
   SCIP_Bool             overestimate,       /**< should the power be overestimated? */
   SCIP_Real             xlb,                /**< lower bound on x, assumed to be non-positive */
   SCIP_Real             xub,                /**< upper bound on x */
   SCIP_Real             xref,               /**< reference point (where to linearize) */
   SCIP_Real             xlbglobal,          /**< global lower bound on x */
   SCIP_Real             xubglobal,          /**< global upper bound on x */
   SCIP_Real*            constant,           /**< buffer to store constant term of estimator */
   SCIP_Real*            slope,              /**< buffer to store slope of estimator */
   SCIP_Bool*            islocal,            /**< buffer to store whether estimator only locally valid, that is, depends on given bounds */
   SCIP_Bool*            success             /**< buffer to store whether estimator could be computed */
)
{
   assert(scip != NULL);
   assert(constant != NULL);
   assert(slope != NULL);
   assert(islocal != NULL);
   assert(success != NULL);
   /* assert((exponent >= 3.0 && EPSISINT((exponent-1.0)/2.0, 0.0)) || exponent > 1.0); <-> exponent > 1.0 */
   assert(exponent >= 1.0);
   assert(xlb < 0.0); /* otherwise estimateParabola should have been called */
   assert(xubglobal <= 0.0 || (root > 0.0 && root < 1.0));

   *success = FALSE;

   /* TODO computeSecant and computeTangent do not know if we are a real signpower (not just an odd power) */

   if( !SCIPisPositive(scip, xub) )
   {
      /* easy case */
      if( !overestimate )
      {
         /* underestimator is secant */
         computeSecant(scip, exponent, xlb, xub, constant, slope, success);
         *islocal = TRUE;
      }
      else
      {
         /* overestimator is tangent */
         computeTangent(scip, exponent, xref, constant, slope, success);

         /* if global upper bound is > 0, then the tangent is only valid locally if the reference point is right of -root*xubglobal */
         *islocal = SCIPisPositive(scip, xubglobal) && xref > -root * xubglobal;
      }
   }
   else
   {
      SCIP_Real c;

      if( !overestimate )
      {
         /* compute the special point which decides between secant and tangent */
         c = -xlb * root;

         if( xref < c )
         {
            /* underestimator is secant between xlb and c */
            computeSecant(scip, exponent, xlb, c, constant, slope, success);
            *islocal = TRUE;
         }
         else
         {
            /* underestimator is tangent */
            computeTangent(scip, exponent, xref, constant, slope, success);

            /* if reference point is left of -root*xlbglobal (c w.r.t. global bounds), then tangent is not valid w.r.t. global bounds */
            *islocal = xref < -root * xlbglobal;
         }
      }
      else
      {
         /* compute the special point which decides between secant and tangent */
         c = -xub * root;

         if( xref <= c )
         {
            /* overestimator is tangent */
            computeTangent(scip, exponent, xref, constant, slope, success);

            /* if reference point is right of -root*xubglobal (c w.r.t. global bounds), then tangent is not valid w.r.t. global bounds */
            *islocal = xref > -root * xubglobal;
         }
         else
         {
            /* overestimator is secant */
            computeSecant(scip, exponent, c, xub, constant, slope, success);
            *islocal = TRUE;
         }
      }
   }
}

/** Separation for positive hyperbola
 *
 * - x^-2, x^-4 with x arbitrary
 * - x^-0.5, x^-1, x^-1.5, x^-3, x^-5 with x >= 0

  5 +----------------------------------------------------------------------+
    |                 +               * +*               +                 |
    |                                 *  *                 x**(-2) ******* |
  4 |-+                               *  *                               +-|
    |                                 *  *                                 |
    |                                 *  *                                 |
    |                                 *  *                                 |
  3 |-+                               *   *                              +-|
    |                                *    *                                |
    |                                *    *                                |
  2 |-+                              *    *                              +-|
    |                                *    *                                |
    |                               *      *                               |
  1 |-+                             *      *                             +-|
    |                               *      *                               |
    |                             **        **                             |
    |                   **********            **********                   |
  0 |*******************                                *******************|
    |                                                                      |
    |                 +                 +                +                 |
 -1 +----------------------------------------------------------------------+
   -10               -5                 0                5                 10

 */
static
void estimateHyperbolaPositive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             exponent,           /**< exponent */
   SCIP_Real             root,               /**< negative root of the polynomial (n-1) y^n - n y^(n-1) + 1, if x has mixed sign (w.r.t. global bounds?) and underestimating */
   SCIP_Bool             overestimate,       /**< should the power be overestimated? */
   SCIP_Real             xlb,                /**< lower bound on x */
   SCIP_Real             xub,                /**< upper bound on x */
   SCIP_Real             xref,               /**< reference point (where to linearize) */
   SCIP_Real             xlbglobal,          /**< global lower bound on x */
   SCIP_Real             xubglobal,          /**< global upper bound on x */
   SCIP_Real*            constant,           /**< buffer to store constant term of estimator */
   SCIP_Real*            slope,              /**< buffer to store slope of estimator */
   SCIP_Bool*            islocal,            /**< buffer to store whether estimator only locally valid, that is, depends on given bounds */
   SCIP_Bool*            success             /**< buffer to store whether estimator could be computed */
)
{
   assert(scip != NULL);
   assert(constant != NULL);
   assert(slope != NULL);
   assert(islocal != NULL);
   assert(success != NULL);
   assert(exponent < 0.0);
   assert(EPSISINT(exponent/2.0, 0.0) || xlb >= 0.0);

   *success = FALSE;

   if( !overestimate )
   {
      if( xlb >= 0.0 || xub <= 0.0 )
      {
         /* underestimate and fixed sign -> tangent */

         if( SCIPisZero(scip, xref) )
         {
            /* estimator would need to have an (essentially) infinite scope
             * first try to make up a better refpoint
             */
            if( xub > 0.0 )
            {
               /* thus xlb >= 0.0; stay close to xlb (probably = 0) */
               if( !SCIPisInfinity(scip, xub) )
                  xref = 0.9 * xlb + 0.1 * xub;
               else
                  xref = 0.1;
            }
            else
            {
               /* xub <= 0.0; stay close to xub (probably = 0) */
               if( !SCIPisInfinity(scip, -xlb) )
                  xref = 0.1 * xlb + 0.9 * xub;
               else
                  xref = 0.1;
            }

            /* if still close to 0, then also bounds are close to 0, then just give up */
            if( SCIPisZero(scip, xref) )
               return;
         }

         computeTangent(scip, exponent, xref, constant, slope, success);

         if( EPSISINT(exponent/2.0, 0.0) )
         {
            /* for even exponents (as in the picture):
             * if x has fixed sign globally, then our tangent is also globally valid
             * however, if x has mixed sign, then it depends on the constellation between reference point and global bounds, whether the tangent is globally valid
             * (see also the longer discussion for the mixed-sign underestimator below )
             */
            if( xref > 0.0 && xlbglobal < 0.0 )
            {
               assert(xubglobal > 0.0);  /* since xref > 0.0 */
               assert(root < 0.0); /* root needs to be given */
               /* if on right side, then tangent is only locally valid if xref is too much to the left */
               *islocal = xref < xlbglobal * root;
            }
            else if( xref < 0.0 && xubglobal > 0.0 )
            {
               assert(xlbglobal < 0.0);  /* since xref < 0.0 */
               assert(root < 0.0); /* root needs to be given */
               /* if on left side, then tangent is only locally valid if xref is too much to the right */
               *islocal = xref > xubglobal * root;
            }
            else
               *islocal = FALSE;
         }
         else
         {
            /* for odd exponents, the tangent is only locally valid if the sign of x is not fixed globally */
            *islocal = xlbglobal * xubglobal < 0.0;
         }
      }
      else
      {
         /* underestimate but mixed sign */
         if( SCIPisInfinity(scip, -xlb) )
         {
            if( SCIPisInfinity(scip, xub) )
            {
               /* underestimator is constant 0, but, wow, that is globally valid */
               *constant = 0.0;
               *slope = 0.0;
               *islocal = FALSE;
               *success = TRUE;
               return;
            }

            /* switch sign of x (mirror on ordinate) to make left bound finite and use its estimator */
            estimateHyperbolaPositive(scip, exponent, root, overestimate, -xub, -xlb, -xref, -xubglobal, -xlbglobal, constant, slope, islocal, success);
            if( *success )
               *slope = -*slope;
         }
         else
         {
            /* The convex envelope of x^exponent for x in [xlb, infinity] is a line (secant) between xlb and some positive
             * coordinate xhat, and x^exponent for x > xhat.
             * Further, on [xlb,xub] with xub < xhat, the convex envelope is the secant between xlb and xub.
             *
             * To find xhat, consider the affine-linear function  l(x) = xlb^n + c * (x - xlb)   where n = exponent
             * we look for a value of x such that f(x) and l(x) coincide and such that l(x) will be tangent to f(x) on that
             * point, that is
             * xhat > 0 such that f(xhat) = l(xhat) and f'(xhat) = l'(xhat)
             * => xhat^n = xlb^n + c * (xhat - xlb)   and   n * xhat^(n-1) = c
             * => xhat^n = xlb^n + n * xhat^n - n * xhat^(n-1) * xlb
             * => 0 = xlb^n + (n-1) * xhat^n - n * xhat^(n-1) * xlb
             *
             * Divide by xlb^n, one gets a polynomial that looks very much like the one for signpower, but a sign is
             * different (since this is *not signed* power):
             * 0 = 1 + (n-1) * y^n - n * y^(n-1)  where y = xhat/xlb
             *
             * The solution y < 0 (because xlb < 0 and we want xhat > 0) is what we expect to be given as "root".
             */
            assert(root < 0.0); /* root needs to be given */
            if( xref <= xlb * root )
            {
               /* If the reference point is left of xhat (=xlb*root), then we can take the
                * secant between xlb and root*xlb (= tangent at root*xlb).
                * However, if xub < root*xlb, then we can tilt the estimator to be the secant between xlb and xub.
                */
               computeSecant(scip, exponent, xlb, MIN(xlb * root, xub), constant, slope, success);
               *islocal = TRUE;
            }
            else
            {
               /* If reference point is right of xhat, then take the tangent at xref.
                * This will still be underestimating for x in [xlb,0], too.
                * The tangent is globally valid, if we had also generated w.r.t. global bounds.
                */
               computeTangent(scip, exponent, xref, constant, slope, success);
               *islocal = xref < xlbglobal * root;
            }
         }
      }
   }
   else
   {
      /* overestimate and mixed sign -> pole is within domain -> cannot overestimate */
      if( xlb < 0.0 && xub > 0.0 )
         return;

      /* overestimate and fixed sign -> secant */
      computeSecant(scip, exponent, xlb, xub, constant, slope, success);
      *islocal = TRUE;
   }

}

/** Separation for mixed-sign hyperbola
 *
 * - x^-1, x^-3, x^-5 without x >= 0 (either x arbitrary or x negative)

    +----------------------------------------------------------------------+
    |                 +                 *                +                 |
  4 |-+                                  *                 x**(-1) *******-|
    |                                    *                                 |
    |                                    *                                 |
    |                                    *                                 |
  2 |-+                                  *                               +-|
    |                                     *                                |
    |                                      **                              |
    |                                        *********                     |
  0 |*********************                            *********************|
    |                     *********                                        |
    |                              **                                      |
    |                                *                                     |
 -2 |-+                               *                                  +-|
    |                                 *                                    |
    |                                 *                                    |
    |                                 *                                    |
 -4 |-+                               *                                  +-|
    |                 +                *+                +                 |
    +----------------------------------------------------------------------+
   -10               -5                 0                5                 10

 */
static
void estimateHyperbolaMixed(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             exponent,           /**< exponent */
   SCIP_Bool             overestimate,       /**< should the power be overestimated? */
   SCIP_Real             xlb,                /**< lower bound on x */
   SCIP_Real             xub,                /**< upper bound on x */
   SCIP_Real             xref,               /**< reference point (where to linearize) */
   SCIP_Real             xlbglobal,          /**< global lower bound on x */
   SCIP_Real             xubglobal,          /**< global upper bound on x */
   SCIP_Real*            constant,           /**< buffer to store constant term of estimator */
   SCIP_Real*            slope,              /**< buffer to store slope of estimator */
   SCIP_Bool*            islocal,            /**< buffer to store whether estimator only locally valid, that is, depends on given bounds */
   SCIP_Bool*            success             /**< buffer to store whether estimator could be computed */
)
{
   assert(scip != NULL);
   assert(constant != NULL);
   assert(slope != NULL);
   assert(islocal != NULL);
   assert(success != NULL);
   assert(exponent < 0.0);
   assert(EPSISINT((exponent-1.0)/2.0, 0.0));
   assert(xlb < 0.0);

   *success = FALSE;

   if( xub <= 0.0 )
   {
      /* x is negative */
      if( !overestimate )
      {
         /* underestimation -> secant */
         computeSecant(scip, exponent, xlb, xub, constant, slope, success);
         *islocal = TRUE;
      }
      else
      {
         /* overestimation -> tangent */

         if( SCIPisZero(scip, xref) && !SCIPisZero(scip, xlb/10.0) )
         {
            /* if xref is very close to 0.0, then slope would be infinite
             * try to move closer to lower bound (if xlb < -10*eps)
             */
            if( !SCIPisInfinity(scip, -xlb) )
               xref = 0.1*xlb + 0.9*xub;
            else
               xref = 0.1;
         }

         computeTangent(scip, exponent, xref, constant, slope, success);
         /* if x does not have a fixed sign globally, then our tangent is not globally valid (power is not convex on global domain) */
         *islocal = xlbglobal * xubglobal < 0.0;
      }
   }
   /* else: x has mixed sign -> pole is within domain -> cannot estimate */
}

/** Separation for roots with exponent in [0,1]
 *
 * - x^0.5 with x >= 0

  8 +----------------------------------------------------------------------+
    |             +             +              +             +             |
  7 |-+                                                     x**0.5 ********|
    |                                                             *********|
    |                                                      ********        |
  6 |-+                                             ********             +-|
    |                                         ******                       |
  5 |-+                                 ******                           +-|
    |                             ******                                   |
    |                        *****                                         |
  4 |-+                  ****                                            +-|
    |               *****                                                  |
  3 |-+          ****                                                    +-|
    |         ***                                                          |
    |      ***                                                             |
  2 |-+  **                                                              +-|
    |  **                                                                  |
  1 |**                                                                  +-|
    |*                                                                     |
    |*            +             +              +             +             |
  0 +----------------------------------------------------------------------+
    0             10            20             30            40            50

 */
static
void estimateRoot(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             exponent,           /**< exponent */
   SCIP_Bool             overestimate,       /**< should the power be overestimated? */
   SCIP_Real             xlb,                /**< lower bound on x */
   SCIP_Real             xub,                /**< upper bound on x */
   SCIP_Real             xref,               /**< reference point (where to linearize) */
   SCIP_Real*            constant,           /**< buffer to store constant term of estimator */
   SCIP_Real*            slope,              /**< buffer to store slope of estimator */
   SCIP_Bool*            islocal,            /**< buffer to store whether estimator only locally valid, that is, depends on given bounds */
   SCIP_Bool*            success             /**< buffer to store whether estimator could be computed */
)
{
   assert(scip != NULL);
   assert(constant != NULL);
   assert(slope != NULL);
   assert(islocal != NULL);
   assert(success != NULL);
   assert(exponent > 0.0);
   assert(exponent < 1.0);
   assert(xlb >= 0.0);

   if( !overestimate )
   {
      /* underestimate -> secant */
      computeSecant(scip, exponent, xlb, xub, constant, slope, success);
      *islocal = TRUE;
   }
   else
   {
      /* overestimate -> tangent */
      if( SCIPisZero(scip, xref) && !SCIPisZero(scip, xub) )
      {
         /* if xref is 0 (then xlb=0 probably), then slope is infinite, then try to move away from 0 */
         if( SCIPisInfinity(scip, xub) )
            xref = 0.9 * xlb + 0.1 * xub;
         else
            xref = 0.1;
      }

      computeTangent(scip, exponent, xref, constant, slope, success);
      *islocal = FALSE;
   }
}


/*
 * Callback methods of expression handler
 */

/** the order of two pow is base_1^expo_1 < base_2^expo_2 if and only if
 * base_1 < base2 or, base_1 = base_2 and expo_1 < expo_2
 */
static
SCIP_DECL_CONSEXPR_EXPRCOMPARE(comparePow)
{  /*lint --e{715}*/
   SCIP_Real expo1;
   SCIP_Real expo2;
   int compareresult;

   compareresult = SCIPcompareConsExprExprs(SCIPgetConsExprExprChildren(expr1)[0],
              SCIPgetConsExprExprChildren(expr2)[0]);
   if( compareresult != 0 )
      return compareresult;

   expo1 = SCIPgetConsExprExprPowExponent(expr1);
   expo2 = SCIPgetConsExprExprPowExponent(expr2);

   return expo1 == expo2 ? 0 : expo1 < expo2 ? -1 : 1; /*lint !e777*/
}

/** simplifies a pow expression.
 * Evaluates the power function when its child is a value expression
 */
static
SCIP_DECL_CONSEXPR_EXPRSIMPLIFY(simplifyPow)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* base;
   SCIP_CONSHDLR* conshdlr;
   SCIP_Real exponent;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(simplifiedexpr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   conshdlr = SCIPfindConshdlr(scip, "expr");
   assert(conshdlr != NULL);

   base = SCIPgetConsExprExprChildren(expr)[0];
   assert(base != NULL);

   exponent = SCIPgetConsExprExprPowExponent(expr);
   /* when exponent is inteer, round exponent so that is actually an integer
    * TODO: should this go in the createConsExprExprPow? */
   if( SCIPisIntegral(scip, exponent) )
      exponent = SCIPround(scip, exponent);

   SCIPdebugPrintf("[simplifyPow] simplifying power with expo %g\n", exponent);

   /* enforces POW1 */
   if( exponent == 0.0 )
   {
      SCIPdebugPrintf("[simplifyPow] POW1\n");
      /* TODO: more checks? */
      if( SCIPgetConsExprExprHdlr(base) == SCIPgetConsExprExprHdlrValue(conshdlr) &&
            SCIPgetConsExprExprValueValue(base) == 0.0 )
      {
         assert(0);
      }
      SCIP_CALL( SCIPcreateConsExprExprValue(scip, conshdlr, simplifiedexpr, 1.0) );
      return SCIP_OKAY;
   }

   /* enforces POW2 */
   if( exponent == 1.0 )
   {
      SCIPdebugPrintf("[simplifyPow] POW2\n");
      *simplifiedexpr = base;
      SCIPcaptureConsExprExpr(*simplifiedexpr);
      return SCIP_OKAY;
   }

   /* enforces POW3 */
   if( SCIPgetConsExprExprHdlr(base) == SCIPgetConsExprExprHdlrValue(conshdlr) )
   {
      SCIP_Real baseval;

      SCIPdebugPrintf("[simplifyPow] POW3\n");
      baseval = SCIPgetConsExprExprValueValue(base);

      /* TODO check if those are all important asserts */
      assert(baseval >= 0.0 || fmod(exponent, 1.0) == 0.0);
      assert(baseval != 0.0 || exponent != 0.0);

      SCIP_CALL( SCIPcreateConsExprExprValue(scip, conshdlr, simplifiedexpr, pow(baseval, exponent)) );
      return SCIP_OKAY;
   }

   /* enforces POW10 */
   if( SCIPgetConsExprExprHdlr(base) == SCIPgetConsExprExprHdlrVar(conshdlr) )
   {
      SCIP_VAR* basevar;

      SCIPdebugPrintf("[simplifyPow] POW10\n");
      basevar = SCIPgetConsExprExprVarVar(base);

      assert(basevar != NULL);

      /* FIXME: if exponent is negative, we could fix the binary variable to 1. However, this is a bit tricky because
       * variables can not be tighten in EXITPRE, where the simplify is also called
       */
      if( SCIPvarIsBinary(basevar) && exponent > 0 )
      {
         *simplifiedexpr = base;
         SCIPcaptureConsExprExpr(*simplifiedexpr);
         return SCIP_OKAY;
      }
   }

   if( SCIPisIntegral(scip, exponent) )
   {
      SCIP_CONSEXPR_EXPR* aux;
      SCIP_CONSEXPR_EXPR* simplifiedaux;

      /* enforces POW5
       * given (pow n (prod 1.0 expr_1 ... expr_k)) we distribute the exponent:
       * -> (prod 1.0 (pow n expr_1) ... (pow n expr_k))
       * notes: - since base is simplified and its coefficient is 1.0 (SP8)
       *        - n is an integer (excluding 1 and 0; see POW1-2 above)
       */
      if( SCIPgetConsExprExprHdlr(base) == SCIPgetConsExprExprHdlrProduct(conshdlr) )
      {
         SCIP_CONSEXPR_EXPR* auxproduct;
         int i;

         /* create empty product */
         SCIP_CALL( SCIPcreateConsExprExprProduct(scip, conshdlr, &auxproduct, 0, NULL, 1.0) );

         for( i = 0; i < SCIPgetConsExprExprNChildren(base); ++i )
         {
            /* create (pow n expr_i) and simplify */
            SCIP_CALL( SCIPcreateConsExprExprPow(scip, conshdlr, &aux,
                     SCIPgetConsExprExprChildren(base)[i], exponent) );
            SCIP_CALL( simplifyPow(scip, aux, &simplifiedaux) );
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &aux) );

            /* append (pow n expr_i) to product */
            SCIP_CALL( SCIPappendConsExprExprProductExpr(scip, auxproduct, simplifiedaux) );
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &simplifiedaux) );
         }

         /* simplify (prod 1.0 (pow n expr_1) ... (pow n expr_k))
          * this calls simplifyProduct directly, since we know its children are simplified */
         SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, auxproduct, simplifiedexpr) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &auxproduct) );
         return SCIP_OKAY;
      }

      /* enforces POW6
       * given (pow n (sum 0.0 coef expr)) we can move `pow` inside `sum`:
       * (pow n (sum 0.0 coef expr) ) -> (sum 0.0 coef^n (pow n expr))
       * notes: - since base is simplified and its constant is 0, then coef != 1.0 (SS7)
       *        - n is an integer (excluding 1 and 0; see POW1-2 above)
       */
      if( SCIPgetConsExprExprHdlr(base) == SCIPgetConsExprExprHdlrSum(conshdlr)
            && SCIPgetConsExprExprNChildren(base) == 1
            && SCIPgetConsExprExprSumConstant(base) == 0.0 )
      {
         SCIP_Real newcoef;

         SCIPdebugPrintf("[simplifyPow] seing a sum with one term, exponent %g\n", exponent);
         /* assert SS7 holds */
         assert(SCIPgetConsExprExprSumCoefs(base)[0] != 1.0);

         /* create (pow n expr) and simplify it
          * note: we call simplifyPow directly, since we know that `expr` is simplified */
         newcoef = pow(SCIPgetConsExprExprSumCoefs(base)[0], exponent);
         SCIP_CALL( SCIPcreateConsExprExprPow(scip, conshdlr, &aux, SCIPgetConsExprExprChildren(base)[0], exponent) );
         SCIP_CALL( simplifyPow(scip, aux, &simplifiedaux) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &aux) );

         /* create (sum (pow n expr)) and simplify it
          * this calls simplifySum directly, since we know its children are simplified */
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, &aux, 1, &simplifiedaux, &newcoef, 0.0) );
         SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, aux, simplifiedexpr) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &aux) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &simplifiedaux) );
         return SCIP_OKAY;
      }

      /* enforces POW7
       * (const + sum alpha_i expr_i)^2 = sum alpha_i^2 expr_i^2
       * + sum_{j < i} 2*alpha_i alpha_j expr_i expr_j
       * + sum const alpha_i expr_i
       * TODO: put some limits on the number of children of the sum being expanded
       */
      if( SCIPgetConsExprExprHdlr(base) == SCIPgetConsExprExprHdlrSum(conshdlr) && exponent == 2 )
      {
         int i;
         int nchildren;
         int nexpandedchildren;
         SCIP_CONSEXPR_EXPR* expansion;
         SCIP_CONSEXPR_EXPR** expandedchildren;
         SCIP_Real* coefs;
         SCIP_Real constant;

         SCIPdebugPrintf("[simplifyPow] expanding sum^%g\n", exponent);

         nchildren = SCIPgetConsExprExprNChildren(base);
         nexpandedchildren = nchildren * (nchildren + 1) / 2 + nchildren;
         SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nexpandedchildren) );
         SCIP_CALL( SCIPallocBufferArray(scip, &expandedchildren, nexpandedchildren) );

         for( i = 0; i < nchildren; ++i )
         {
            int j;
            SCIP_CONSEXPR_EXPR* expansionchild;
            SCIP_CONSEXPR_EXPR* prodchildren[2];
            prodchildren[0] = SCIPgetConsExprExprChildren(base)[i];

            /* create and simplify expr_i * expr_j */
            for( j = 0; j < i; ++j )
            {
               prodchildren[1] = SCIPgetConsExprExprChildren(base)[j];
               coefs[i*(i+1)/2 + j] = 2 * SCIPgetConsExprExprSumCoefs(base)[i] * SCIPgetConsExprExprSumCoefs(base)[j];

               SCIP_CALL( SCIPcreateConsExprExprProduct(scip, conshdlr, &expansionchild, 2, prodchildren, 1.0) );
               SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, expansionchild, &expandedchildren[i*(i+1)/2 + j]) ); /* this call simplifyProduct */
               SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expansionchild) );
            }
            /* create and simplify expr_i * expr_i */
            prodchildren[1] = SCIPgetConsExprExprChildren(base)[i];
            coefs[i*(i+1)/2 + i] = SCIPgetConsExprExprSumCoefs(base)[i] * SCIPgetConsExprExprSumCoefs(base)[i];

            SCIP_CALL( SCIPcreateConsExprExprProduct(scip, conshdlr, &expansionchild, 2, prodchildren, 1.0) );
            SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, expansionchild, &expandedchildren[i*(i+1)/2 + i]) ); /* this call simplifyProduct */
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expansionchild) );
         }
         /* create const * alpha_i expr_i */
         for( i = 0; i < nchildren; ++i )
         {
            coefs[i + nexpandedchildren - nchildren] = 2 * SCIPgetConsExprExprSumConstant(base) * SCIPgetConsExprExprSumCoefs(base)[i];
            expandedchildren[i + nexpandedchildren - nchildren] = SCIPgetConsExprExprChildren(base)[i];
         }

         constant = SCIPgetConsExprExprSumConstant(base);
         constant *= constant;
         /* create sum of all the above and simplify it with simplifySum since all of its children are simplified! */
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, &expansion, nexpandedchildren,
                  expandedchildren, coefs, constant) );
         SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, expansion, simplifiedexpr) ); /* this calls simplifySum */

         /* release eveything */
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expansion) );
         /* release the *created* expanded children */
         for( i = 0; i < nexpandedchildren - nchildren; ++i )
         {
            SCIP_CALL( SCIPreleaseConsExprExpr(scip, &expandedchildren[i]) );
         }
         SCIPfreeBufferArray(scip, &expandedchildren);
         SCIPfreeBufferArray(scip, &coefs);

         return SCIP_OKAY;
      }

      /* enforces POW8
       * given (pow n (pow expo expr)) we distribute the exponent:
       * -> (pow n*expo expr)
       * notes: n is an integer (excluding 1 and 0; see POW1-2 above)
       */
      /* FIXME: use SCIPgetConsExprExprHdlrPow */
      if( strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(base)), "pow") == 0 )
      {
         SCIP_Real newexponent;

         newexponent = SCIPgetConsExprExprPowExponent(base) * exponent;
         SCIP_CALL( SCIPcreateConsExprExprPow(scip, conshdlr, &aux, SCIPgetConsExprExprChildren(base)[0], newexponent) );
         SCIP_CALL( simplifyPow(scip, aux, simplifiedexpr) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &aux) );

         return SCIP_OKAY;
      }
   }
   else
   {
      /* enforces POW9
       *
       * FIXME code of POW6 is very similar
       */
      if( SCIPgetConsExprExprNChildren(base) == 1
         && SCIPgetConsExprExprHdlr(base) == SCIPgetConsExprExprHdlrSum(conshdlr)
         && SCIPgetConsExprExprSumConstant(base) == 0.0
         && SCIPgetConsExprExprSumCoefs(base)[0] >= 0.0 )
      {
         SCIP_CONSEXPR_EXPR* simplifiedaux;
         SCIP_CONSEXPR_EXPR* aux;
         SCIP_Real newcoef;

         SCIPdebugPrintf("[simplifyPow] seing a sum with one term, exponent %g\n", exponent);
         /* assert SS7 holds */
         assert(SCIPgetConsExprExprSumCoefs(base)[0] != 1.0);

         /* create (pow n expr) and simplify it
          * note: we call simplifyPow directly, since we know that `expr` is simplified */
         SCIP_CALL( SCIPcreateConsExprExprPow(scip, conshdlr, &aux, SCIPgetConsExprExprChildren(base)[0], exponent) );
         SCIP_CALL( simplifyPow(scip, aux, &simplifiedaux) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &aux) );

         /* create (sum (pow n expr)) and simplify it
          * this calls simplifySum directly, since we know its child is simplified! */
         newcoef = pow(SCIPgetConsExprExprSumCoefs(base)[0], exponent);
         SCIP_CALL( SCIPcreateConsExprExprSum(scip, conshdlr, &aux, 1, &simplifiedaux, &newcoef, 0.0) );
         SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, aux, simplifiedexpr) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &aux) );
         SCIP_CALL( SCIPreleaseConsExprExpr(scip, &simplifiedaux) );

         return SCIP_OKAY;
      }
   }

   SCIPdebugPrintf("[simplifyPow] power is simplified\n");
   *simplifiedexpr = expr;

   /* we have to capture it, since it must simulate a "normal" simplified call in which a new expression is created */
   SCIPcaptureConsExprExpr(*simplifiedexpr);

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRCOPYHDLR(copyhdlrPow)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPincludeConsExprExprHdlrPow(scip, consexprhdlr) );
   *valid = TRUE;

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRCOPYDATA(copydataPow)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRDATA* sourceexprdata;

   assert(targetexprdata != NULL);
   assert(sourceexpr != NULL);

   sourceexprdata = SCIPgetConsExprExprData(sourceexpr);
   assert(sourceexprdata != NULL);

   *targetexprdata = NULL;

   SCIP_CALL( createData(targetscip, targetexprdata, sourceexprdata->exponent) );

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRFREEDATA(freedataPow)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRDATA* exprdata;

   assert(expr != NULL);

   exprdata = SCIPgetConsExprExprData(expr);
   assert(exprdata != NULL);

   SCIPfreeBlockMemory(scip, &exprdata);
   SCIPsetConsExprExprData(expr, NULL);

   return SCIP_OKAY;
}

/** @todo: use precedence for better printing */
static
SCIP_DECL_CONSEXPR_EXPRPRINT(printPow)
{  /*lint --e{715}*/
   assert(expr != NULL);

   switch( stage )
   {
      case SCIP_CONSEXPRITERATOR_ENTEREXPR :
      {
         /* print function with opening parenthesis */
         SCIPinfoMessage(scip, file, "(");
         break;
      }

      case SCIP_CONSEXPRITERATOR_VISITINGCHILD :
      {
         assert(currentchild == 0);
         break;
      }

      case SCIP_CONSEXPRITERATOR_LEAVEEXPR :
      {

         SCIP_Real exponent = SCIPgetConsExprExprPowExponent(expr);

         /* print closing parenthesis */
         if( exponent >= 0.0 )
            SCIPinfoMessage(scip, file, ")^%g", exponent);
         else
            SCIPinfoMessage(scip, file, ")^(%g)", exponent);

         break;
      }

      case SCIP_CONSEXPRITERATOR_VISITEDCHILD :
      default: ;
   }

   return SCIP_OKAY;
}

/** expression point evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPREVAL(evalPow)
{  /*lint --e{715}*/
   SCIP_Real exponent;
   SCIP_Real base;

   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(SCIPgetConsExprExprValue(SCIPgetConsExprExprChildren(expr)[0]) != SCIP_INVALID); /*lint !e777*/

   exponent = SCIPgetConsExprExprPowExponent(expr);
   base = SCIPgetConsExprExprValue(SCIPgetConsExprExprChildren(expr)[0]);

   *val = pow(base, exponent);

   /* if there is a domain, pole, or range error, pow() should return some kind of NaN, infinity, or HUGE_VAL
    * we could also work with floating point exceptions or errno, but I am not sure this would be thread-safe
    */
   if( !SCIPisFinite(*val) || *val == HUGE_VAL || *val == -HUGE_VAL )
      *val = SCIP_INVALID;

   return SCIP_OKAY;
}

/** expression derivative evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRBWDIFF(bwdiffPow)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_Real exponent;

   assert(expr != NULL);
   assert(SCIPgetConsExprExprData(expr) != NULL);
   assert(childidx == 0);
   assert(SCIPgetConsExprExprValue(expr) != SCIP_INVALID); /*lint !e777*/

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(child)), "val") != 0);

   exponent = SCIPgetConsExprExprPowExponent(expr);
   assert(exponent != 1.0 && exponent != 0.0);

   /* x^exponent is not differentiable for x = 0 and exponent in ]0,1[ */
   if( exponent > 0.0 && exponent < 1.0 && SCIPgetConsExprExprValue(child) == 0.0 )
      *val = SCIP_INVALID;
   else
      *val = exponent * pow(SCIPgetConsExprExprValue(child), exponent - 1.0);

   return SCIP_OKAY;
}

/** expression interval evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRINTEVAL(intevalPow)
{  /*lint --e{715}*/
   SCIP_INTERVAL childinterval;
   SCIP_Real exponent;

   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   childinterval = SCIPgetConsExprExprActivity(scip, SCIPgetConsExprExprChildren(expr)[0]);
   assert(!SCIPintervalIsEmpty(SCIP_INTERVAL_INFINITY, childinterval));

   exponent = SCIPgetConsExprExprPowExponent(expr);

   SCIPintervalPowerScalar(SCIP_INTERVAL_INFINITY, interval, childinterval, exponent);

   return SCIP_OKAY;
}

/** expression estimator callback */
static
SCIP_DECL_CONSEXPR_EXPRESTIMATE(estimatePow)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPRDATA* exprdata;
   SCIP_CONSEXPR_EXPR* child;
   SCIP_VAR* childvar;
   SCIP_Real childlb;
   SCIP_Real childub;
   SCIP_Real exponent;
   SCIP_Real refpoint;
   SCIP_Bool isinteger;
   SCIP_Bool iseven;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), "expr") == 0);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), EXPRHDLR_NAME) == 0);
   assert(coefs != NULL);
   assert(constant != NULL);
   assert(islocal != NULL);
   assert(success != NULL);

   *success = FALSE;

   /* get aux variables: we over- or underestimate childvar^exponent  */
   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);
   childvar = SCIPgetConsExprExprAuxVar(child);
   assert(childvar != NULL);

   refpoint = SCIPgetSolVal(scip, sol, childvar);

   SCIPdebugMsg(scip, "%sestimation of x^%g at x=%g\n", overestimate ? "over" : "under", SCIPgetConsExprExprData(expr)->exponent, refpoint);

   /* we can not generate a cut at +/- infinity */
   if( SCIPisInfinity(scip, REALABS(refpoint)) )
      return SCIP_OKAY;

   childlb = SCIPvarGetLbLocal(childvar);
   childub = SCIPvarGetUbLocal(childvar);

   /* if child is essentially constant, then there should be no point in separation */
   if( SCIPisEQ(scip, childlb, childub) ) /* @todo maybe return a constant estimator? */
      return SCIP_OKAY;

   exprdata = SCIPgetConsExprExprData(expr);
   exponent = exprdata->exponent;
   assert(exponent != 1.0 && exponent != 0.0); /* this should have been simplified */

   isinteger = EPSISINT(exponent, 0.0);
   iseven = isinteger && EPSISINT(exponent/2.0, 0.0);

   /* adjust the reference point */
   refpoint = SCIPisLT(scip, refpoint, childlb) ? childlb : refpoint;
   refpoint = SCIPisGT(scip, refpoint, childub) ? childub : refpoint;
   assert(SCIPisLE(scip, refpoint, childub) && SCIPisGE(scip, refpoint, childlb));

   /* if exponent is not integral, then child must be non-negative */
   if( !isinteger && childlb < 0.0 )
   {
      /* somewhere we should have tightened the bound on x, but small tightening are not always applied by SCIP
       * it is ok to do this tightening here, but let's assert that we were close to 0.0 already
       */
      assert(SCIPisFeasZero(scip, childlb));
      childlb = 0.0;
      refpoint = MAX(refpoint, 0.0);
   }
   assert(isinteger || childlb >= 0.0);

   if( exponent == 2.0 )
   {
      /* initialize, because SCIPaddSquareXyz only adds to existing values */
      *success = TRUE;
      *coefs = 0.0;
      *constant = 0.0;

      /* important special case: quadratic case */
      if( overestimate )
      {
         SCIPaddSquareSecant(scip, 1.0, childlb, childub, refpoint, coefs, constant, success);
         *islocal = TRUE; /* secants are only valid locally */
      }
      else
      {
         SCIPaddSquareLinearization(scip, 1.0, refpoint, SCIPvarIsIntegral(childvar), coefs, constant, success);
         *islocal = FALSE; /* linearizations are globally valid */
      }
   }
   else if( exponent > 0.0 && iseven )
   {
      estimateParabola(scip, exponent, overestimate, childlb, childub, refpoint, constant, coefs, islocal, success);
   }
   else if( exponent > 1.0 && childlb >= 0.0 )
   {
      /* FIXME tangents on parabola are only globally valid if global lower bound is also >= 0.0 (thus not signpower) */
      estimateParabola(scip, exponent, overestimate, childlb, childub, refpoint, constant, coefs, islocal, success);
   }
   else if( exponent > 1.0 )  /* and !iseven && childlb < 0.0 due to previous if */
   {
      /* compute root if not known yet; only needed if mixed sign (global child ub > 0) */
      if( exprdata->root == SCIP_INVALID && SCIPvarGetUbGlobal(childvar) > 0.0 ) /*lint !e777*/
      {
         SCIP_CALL( computeSignpowerRoot(scip, &exprdata->root, exponent) );
      }
      estimateSignpower(scip, exponent, exprdata->root, overestimate, childlb, childub, refpoint,
            SCIPvarGetLbGlobal(childvar), SCIPvarGetUbGlobal(childvar), constant, coefs, islocal, success);
   }
   else if( exponent < 0.0 && (iseven || childlb >= 0.0) )
   {
      /* compute root if not known yet; only needed if mixed sign (globally) and iseven */
      if( exprdata->root == SCIP_INVALID && iseven ) /*lint !e777*/
      {
         SCIP_CALL( computeHyperbolaRoot(scip, &exprdata->root, exponent) );
      }
      estimateHyperbolaPositive(scip, exponent, exprdata->root, overestimate, childlb, childub, refpoint, SCIPvarGetLbGlobal(childvar), SCIPvarGetUbGlobal(childvar), constant, coefs, islocal, success);
   }
   else if( exponent < 0.0 )
   {
      assert(!iseven); /* should hold due to previous if */
      assert(childlb < 0.0); /* should hold due to previous if */
      assert(isinteger); /* should hold because childlb < 0.0 (same as assert above) */

      estimateHyperbolaMixed(scip, exponent, overestimate, childlb, childub, refpoint, SCIPvarGetLbGlobal(childvar), SCIPvarGetUbGlobal(childvar), constant, coefs, islocal, success);
   }
   else
   {
      assert(exponent < 1.0); /* the only case that should be left */
      assert(exponent > 0.0); /* should hold due to previous if */

      estimateRoot(scip, exponent, overestimate, childlb, childub, refpoint, constant, coefs, islocal, success);
   }

   return SCIP_OKAY;
}

/** expression reverse propagaton callback */
static
SCIP_DECL_CONSEXPR_EXPRREVERSEPROP(reversepropPow)
{  /*lint --e{715}*/
   SCIP_INTERVAL interval;
   SCIP_INTERVAL child;
   SCIP_Real exponent;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(nreductions != NULL);

   *nreductions = 0;

   exponent = SCIPgetConsExprExprPowExponent(expr);

   interval = SCIPgetConsExprExprActivity(scip, expr);
   child = SCIPgetConsExprExprActivity(scip, SCIPgetConsExprExprChildren(expr)[0]);

   SCIPdebugMsg(scip, "reverseprop x^%g in [%g,%g], x = [%g,%g]", exponent, interval.inf, interval.sup, child.inf, child.sup);

   if( SCIPintervalIsEntire(SCIP_INTERVAL_INFINITY, interval) )
   {
      /* if exponent is not integral, then make sure that child is non-negative */
      if( !EPSISINT(exponent, 0.0) && child.inf < 0.0 )
      {
         SCIPintervalSetBounds(&interval, 0.0, child.sup);
      }
      else
      {
         SCIPdebugMsgPrint(scip, "-> no improvement\n");
         return SCIP_OKAY;
      }
   }
   else
   {
      /* f = pow(c0, alpha) -> c0 = pow(f, 1/alpha) */
      SCIPintervalPowerScalarInverse(SCIP_INTERVAL_INFINITY, &interval, child, exponent, interval);
   }

   SCIPdebugMsgPrint(scip, " -> [%g,%g]\n", interval.inf, interval.sup);

   /* try to tighten the bounds of the child node */
   SCIP_CALL( SCIPtightenConsExprExprInterval(scip, SCIPgetConsExprExprChildren(expr)[0], interval, force, reversepropqueue, infeasible,
         nreductions) );

   return SCIP_OKAY;
}

/** expression hash callback */
static
SCIP_DECL_CONSEXPR_EXPRHASH(hashPow)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(hashkey != NULL);
   assert(childrenhashes != NULL);

   *hashkey = EXPRHDLR_HASHKEY;
   *hashkey ^= childrenhashes[0];

   return SCIP_OKAY;
}

/** expression curvature detection callback */
static
SCIP_DECL_CONSEXPR_EXPRCURVATURE(curvaturePow)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_EXPRCURV childcurv;
   SCIP_INTERVAL childinterval;
   SCIP_Real exponent;
   SCIP_Bool expisint;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(curvature != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   exponent = SCIPgetConsExprExprPowExponent(expr);
   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);
   childcurv = SCIPgetConsExprExprCurvature(child);
   childinterval = SCIPgetConsExprExprActivity(scip, child);

   *curvature = SCIP_EXPRCURV_UNKNOWN;

   assert(childinterval.inf <= childinterval.sup);

   if( exponent == 0.0 )
   {
      *curvature = SCIP_EXPRCURV_LINEAR;
      return SCIP_OKAY;
   }

   if( exponent == 1.0 )
   {
      *curvature = childcurv;
      return SCIP_OKAY;
   }

   expisint = EPSISINT(exponent, 0.0); /*lint !e835*/

   /* if exponent is fractional, then power is not defined for a negative base
    * thus, consider only positive part of basebounds
    */
   if( !expisint && childinterval.inf < 0.0 )
   {
      childinterval.inf = 0.0;
      if( childinterval.sup < 0.0 )
      {
         *curvature = SCIP_EXPRCURV_LINEAR;
         return SCIP_OKAY;
      }
   }

   /* if basebounds contains 0.0, consider negative and positive interval separately, if possible */
   if( childinterval.inf < 0.0 && childinterval.sup > 0.0 )
   {
      SCIP_INTERVAL leftbounds;
      SCIP_INTERVAL rightbounds;

      /* something like x^(-2) may look convex on each side of zero, but is not convex on the whole interval due to the singularity at 0.0 */
      if( exponent < 0.0 )
      {
         *curvature = SCIP_EXPRCURV_UNKNOWN;
         return SCIP_OKAY;
      }

      SCIPintervalSetBounds(&leftbounds,  childinterval.inf, 0.0);
      SCIPintervalSetBounds(&rightbounds, 0.0, childinterval.sup);

      *curvature = (SCIP_EXPRCURV) (SCIPexprcurvPower(leftbounds,  childcurv, exponent) & SCIPexprcurvPower(rightbounds, childcurv, exponent));
      return SCIP_OKAY;
   }
   assert(childinterval.inf >= 0.0 || childinterval.sup <= 0.0);

   /* (base^exponent)'' = exponent * ( (exponent-1) base^(exponent-2) (base')^2 + base^(exponent-1) base'' )
    *
    * if base'' is positive, i.e., base is convex, then
    * - for base > 0.0 and exponent > 1.0, the second deriv. is positive -> convex
    * - for base < 0.0 and exponent > 1.0, we can't say (first and second summand opposite signs)
    * - for base > 0.0 and 0.0 < exponent < 1.0, we can't say (first sommand negative, second summand positive)
    * - for base > 0.0 and exponent < 0.0, we can't say (first and second summand opposite signs)
    * - for base < 0.0 and exponent < 0.0 and even, the second deriv. is positive -> convex
    * - for base < 0.0 and exponent < 0.0 and odd, the second deriv. is negative -> concave
    *
    * if base'' is negative, i.e., base is concave, then
    * - for base > 0.0 and exponent > 1.0, we can't say (first summand positive, second summand negative)
    * - for base < 0.0 and exponent > 1.0 and even, the second deriv. is positive -> convex
    * - for base < 0.0 and exponent > 1.0 and odd, the second deriv. is negative -> concave
    * - for base > 0.0 and 0.0 < exponent < 1.0, the second deriv. is negative -> concave
    * - for base > 0.0 and exponent < 0.0, the second deriv. is positive -> convex
    * - for base < 0.0 and exponent < 0.0, we can't say (first and second summand opposite signs)
    *
    * if base'' is zero, i.e., base is linear, then
    *   (base^exponent)'' = exponent * (exponent-1) base^(exponent-2) (base')^2
    * - just multiply signs
    */

   if( childcurv == SCIP_EXPRCURV_LINEAR )
   {
      SCIP_Real sign;

      /* base^(exponent-2) is negative, if base < 0.0 and exponent is odd */
      sign = exponent * (exponent - 1.0);
      assert(childinterval.inf >= 0.0 || expisint);
      if( childinterval.inf < 0.0 && ((int)exponent)%2 != 0 )
         sign *= -1.0;
      assert(sign != 0.0);

      *curvature =  sign > 0.0 ? SCIP_EXPRCURV_CONVEX : SCIP_EXPRCURV_CONCAVE;
      return SCIP_OKAY;
   }

   if( childcurv == SCIP_EXPRCURV_CONVEX )
   {
      if( childinterval.sup <= 0.0 && exponent < 0.0 && expisint )
         *curvature = ((int)exponent)%2 == 0 ? SCIP_EXPRCURV_CONVEX : SCIP_EXPRCURV_CONCAVE;
      if( childinterval.inf >= 0.0 && exponent > 1.0 )
         *curvature = SCIP_EXPRCURV_CONVEX ;
      return SCIP_OKAY;
   }

   if( childcurv == SCIP_EXPRCURV_CONCAVE )
   {
      if( childinterval.sup <= 0.0 && exponent > 1.0 && expisint )
         *curvature = ((int)exponent)%2 == 0 ? SCIP_EXPRCURV_CONVEX : SCIP_EXPRCURV_CONCAVE;
      if( childinterval.inf >= 0.0 && exponent < 1.0 )
         *curvature = exponent < 0.0 ? SCIP_EXPRCURV_CONVEX : SCIP_EXPRCURV_CONCAVE;
      return SCIP_OKAY;
   }

   return SCIP_OKAY;
}

/** expression monotonicity detection callback */
static
SCIP_DECL_CONSEXPR_EXPRMONOTONICITY(monotonicityPow)
{  /*lint --e{715}*/
   SCIP_INTERVAL interval;
   SCIP_Real exponent;
   SCIP_Real inf;
   SCIP_Real sup;
   SCIP_Bool expisint;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(result != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);
   assert(childidx == 0);

   assert(SCIPgetConsExprExprChildren(expr)[0] != NULL);
   interval = SCIPgetConsExprExprActivity(scip, SCIPgetConsExprExprChildren(expr)[0]);

   *result = SCIP_MONOTONE_UNKNOWN;
   inf = SCIPintervalGetInf(interval);
   sup = SCIPintervalGetSup(interval);
   exponent = SCIPgetConsExprExprPowExponent(expr);
   expisint = EPSISINT(exponent, 0.0); /*lint !e835*/

   if( expisint )
   {
      SCIP_Bool expisodd = ceil(exponent/2) != exponent/2; /*lint !e777*/

      if( expisodd )
      {
         /* x^1, x^3, ... */
         if( exponent >= 0.0 )
            *result = SCIP_MONOTONE_INC;

         /* ..., x^-3, x^-1 are decreasing if 0 is not in ]inf,sup[ */
         else if( inf >= 0.0 || sup <= 0.0 )
            *result = SCIP_MONOTONE_DEC;
      }
      /* ..., x^-4, x^-2, x^2, x^4, ... */
      else
      {
         /* function is not monotone if 0 is in ]inf,sup[ */
         if( inf >= 0.0 )
            *result = exponent >= 0.0 ? SCIP_MONOTONE_INC : SCIP_MONOTONE_DEC;
         else if( sup <= 0.0 )
            *result = exponent >= 0.0 ? SCIP_MONOTONE_DEC : SCIP_MONOTONE_INC;
      }
   }
   else
   {
      /* note that the expression is not defined for negative input values
       *
       * - increasing iff exponent >= 0
       * - decreasing iff exponent <= 0
       */
      *result = exponent >= 0.0 ? SCIP_MONOTONE_INC : SCIP_MONOTONE_DEC;
   }

   return SCIP_OKAY;
}

/** expression integrality detection callback */
static
SCIP_DECL_CONSEXPR_EXPRINTEGRALITY(integralityPow)
{  /*lint --e{715}*/
   SCIP_CONSEXPR_EXPR* child;
   SCIP_Real exponent;
   SCIP_Bool expisint;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(isintegral != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 1);

   *isintegral = FALSE;

   child = SCIPgetConsExprExprChildren(expr)[0];
   assert(child != NULL);

   /* expression can not be integral if child is not */
   if( !SCIPisConsExprExprIntegral(child) )
      return SCIP_OKAY;

   exponent = SCIPgetConsExprExprPowExponent(expr);
   assert(exponent != 0.0);
   expisint = EPSISINT(exponent, 0.0); /*lint !e835*/

   /* expression is integral if and only if exponent non-negative and integral */
   *isintegral = expisint && exponent >= 0.0;

   return SCIP_OKAY;
}

/** creates the handler for power expression and includes it into the expression constraint handler */
SCIP_RETCODE SCIPincludeConsExprExprHdlrPow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;

   SCIP_CALL( SCIPincludeConsExprExprHdlrBasic(scip, consexprhdlr, &exprhdlr, EXPRHDLR_NAME, EXPRHDLR_DESC,
         EXPRHDLR_PRECEDENCE, evalPow, NULL) );
   assert(exprhdlr != NULL);

   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeHdlr(scip, consexprhdlr, exprhdlr, copyhdlrPow, NULL) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeData(scip, consexprhdlr, exprhdlr, copydataPow, freedataPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrSimplify(scip, consexprhdlr, exprhdlr, simplifyPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrPrint(scip, consexprhdlr, exprhdlr, printPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrIntEval(scip, consexprhdlr, exprhdlr, intevalPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrSepa(scip, consexprhdlr, exprhdlr, NULL, NULL, NULL, estimatePow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrReverseProp(scip, consexprhdlr, exprhdlr, reversepropPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrHash(scip, consexprhdlr, exprhdlr, hashPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCompare(scip, consexprhdlr, exprhdlr, comparePow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrBwdiff(scip, consexprhdlr, exprhdlr, bwdiffPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCurvature(scip, consexprhdlr, exprhdlr, curvaturePow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrMonotonicity(scip, consexprhdlr, exprhdlr, monotonicityPow) );
   SCIP_CALL( SCIPsetConsExprExprHdlrIntegrality(scip, consexprhdlr, exprhdlr, integralityPow) );

   return SCIP_OKAY;
}

/** creates a power expression */
SCIP_RETCODE SCIPcreateConsExprExprPow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR**  expr,               /**< pointer where to store expression */
   SCIP_CONSEXPR_EXPR*   child,              /**< single child */
   SCIP_Real             exponent            /**< exponent of the power expression */
   )
{
   SCIP_CONSEXPR_EXPRDATA* exprdata;

   assert(expr != NULL);
   assert(child != NULL);
   assert(SCIPfindConsExprExprHdlr(consexprhdlr, EXPRHDLR_NAME) != NULL);

   SCIP_CALL( createData(scip, &exprdata, exponent) );
   assert(exprdata != NULL);

   SCIP_CALL( SCIPcreateConsExprExpr(scip, expr, SCIPfindConsExprExprHdlr(consexprhdlr, EXPRHDLR_NAME), exprdata, 1, &child) );

   return SCIP_OKAY;
}

/** gets the exponent of a power expression */
SCIP_Real SCIPgetConsExprExprPowExponent(
   SCIP_CONSEXPR_EXPR*   expr                /**< expression */
   )
{
   SCIP_CONSEXPR_EXPRDATA* exprdata;

   assert(expr != NULL);

   exprdata = SCIPgetConsExprExprData(expr);
   assert(exprdata != NULL);

   return exprdata->exponent;
}
