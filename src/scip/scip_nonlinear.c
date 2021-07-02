/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2021 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   scip_nonlinear.c
 * @ingroup OTHER_CFILES
 * @brief  public methods for nonlinear functions
 * @author Tobias Achterberg
 * @author Timo Berthold
 * @author Gerald Gamrath
 * @author Leona Gottwald
 * @author Stefan Heinz
 * @author Gregor Hendel
 * @author Thorsten Koch
 * @author Alexander Martin
 * @author Marc Pfetsch
 * @author Michael Winkler
 * @author Kati Wolter
 *
 * @todo check all SCIP_STAGE_* switches, and include the new stages TRANSFORMED and INITSOLVE
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#define _USE_MATH_DEFINES   /* to get M_E on Windows */  /*lint !750 */

#include "blockmemshell/memory.h"
#include "scip/expr_varidx.h"
#include "scip/scip_expr.h"
#include "scip/pub_expr.h"
#include "scip/dbldblarith.h"
#include "scip/pub_lp.h"
#include "scip/pub_message.h"
#include "scip/pub_misc.h"
#include "scip/pub_nlp.h"
#include "scip/pub_var.h"
#include "scip/scip_nlpi.h"
#include "scip/scip_mem.h"
#include "scip/scip_message.h"
#include "scip/scip_nonlinear.h"
#include "scip/scip_numerics.h"
#include "scip/scip_prob.h"

/** computes coefficients of linearization of a square term in a reference point */
void SCIPaddSquareLinearization(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             sqrcoef,            /**< coefficient of square term */
   SCIP_Real             refpoint,           /**< point where to linearize */
   SCIP_Bool             isint,              /**< whether corresponding variable is a discrete variable, and thus linearization could be moved */
   SCIP_Real*            lincoef,            /**< buffer to add coefficient of linearization */
   SCIP_Real*            linconstant,        /**< buffer to add constant of linearization */
   SCIP_Bool*            success             /**< buffer to set to FALSE if linearization has failed due to large numbers */
   )
{
   assert(scip != NULL);
   assert(lincoef != NULL);
   assert(linconstant != NULL);
   assert(success != NULL);

   if( sqrcoef == 0.0 )
      return;

   if( SCIPisInfinity(scip, REALABS(refpoint)) )
   {
      *success = FALSE;
      return;
   }

   if( !isint || SCIPisIntegral(scip, refpoint) )
   {
      SCIP_Real tmp;

      /* sqrcoef * x^2  ->  tangent in refpoint = sqrcoef * 2 * refpoint * (x - refpoint) */

      tmp = sqrcoef * refpoint;

      if( SCIPisInfinity(scip, 2.0 * REALABS(tmp)) )
      {
         *success = FALSE;
         return;
      }

      *lincoef += 2.0 * tmp;
      tmp *= refpoint;
      *linconstant -= tmp;
   }
   else
   {
      /* sqrcoef * x^2 -> secant between f=floor(refpoint) and f+1 = sqrcoef * (f^2 + ((f+1)^2 - f^2) * (x-f))
       * = sqrcoef * (-f*(f+1) + (2*f+1)*x)
       */
      SCIP_Real f;
      SCIP_Real coef;
      SCIP_Real constant;

      f = SCIPfloor(scip, refpoint);

      coef     =  sqrcoef * (2.0 * f + 1.0);
      constant = -sqrcoef * f * (f + 1.0);

      if( SCIPisInfinity(scip, REALABS(coef)) || SCIPisInfinity(scip, REALABS(constant)) )
      {
         *success = FALSE;
         return;
      }

      *lincoef     += coef;
      *linconstant += constant;
   }
}

/** computes coefficients of secant of a square term */
void SCIPaddSquareSecant(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             sqrcoef,            /**< coefficient of square term */
   SCIP_Real             lb,                 /**< lower bound on variable */
   SCIP_Real             ub,                 /**< upper bound on variable */
   SCIP_Real*            lincoef,            /**< buffer to add coefficient of secant */
   SCIP_Real*            linconstant,        /**< buffer to add constant of secant */
   SCIP_Bool*            success             /**< buffer to set to FALSE if secant has failed due to large numbers or unboundedness */
   )
{
   SCIP_Real coef;
   SCIP_Real constant;

   assert(scip != NULL);
   assert(!SCIPisInfinity(scip,  lb));
   assert(!SCIPisInfinity(scip, -ub));
   assert(SCIPisLE(scip, lb, ub));
   assert(lincoef != NULL);
   assert(linconstant != NULL);
   assert(success != NULL);

   if( sqrcoef == 0.0 )
      return;

   if( SCIPisInfinity(scip, -lb) || SCIPisInfinity(scip, ub) )
   {
      /* unboundedness */
      *success = FALSE;
      return;
   }

   /* sqrcoef * x^2 -> sqrcoef * (lb * lb + (ub*ub - lb*lb)/(ub-lb) * (x-lb)) = sqrcoef * (lb*lb + (ub+lb)*(x-lb))
    *  = sqrcoef * ((lb+ub)*x - lb*ub)
    */
   coef     =  sqrcoef * (lb + ub);
   constant = -sqrcoef * lb * ub;
   if( SCIPisInfinity(scip, REALABS(coef)) || SCIPisInfinity(scip, REALABS(constant)) )
   {
      *success = FALSE;
      return;
   }

   *lincoef     += coef;
   *linconstant += constant;
}

/** computes coefficients of linearization of a bilinear term in a reference point */
void SCIPaddBilinLinearization(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             bilincoef,          /**< coefficient of bilinear term */
   SCIP_Real             refpointx,          /**< point where to linearize first  variable */
   SCIP_Real             refpointy,          /**< point where to linearize second variable */
   SCIP_Real*            lincoefx,           /**< buffer to add coefficient of first  variable in linearization */
   SCIP_Real*            lincoefy,           /**< buffer to add coefficient of second variable in linearization */
   SCIP_Real*            linconstant,        /**< buffer to add constant of linearization */
   SCIP_Bool*            success             /**< buffer to set to FALSE if linearization has failed due to large numbers */
   )
{
   SCIP_Real constant;

   assert(scip != NULL);
   assert(lincoefx != NULL);
   assert(lincoefy != NULL);
   assert(linconstant != NULL);
   assert(success != NULL);

   if( bilincoef == 0.0 )
      return;

   if( SCIPisInfinity(scip, REALABS(refpointx)) || SCIPisInfinity(scip, REALABS(refpointy)) )
   {
      *success = FALSE;
      return;
   }

   /* bilincoef * x * y ->  bilincoef * (refpointx * refpointy + refpointy * (x - refpointx) + refpointx * (y - refpointy))
    *                    = -bilincoef * refpointx * refpointy + bilincoef * refpointy * x + bilincoef * refpointx * y
    */

   constant = -bilincoef * refpointx * refpointy;

   if( SCIPisInfinity(scip, REALABS(bilincoef * refpointx)) || SCIPisInfinity(scip, REALABS(bilincoef * refpointy))
      || SCIPisInfinity(scip, REALABS(constant)) )
   {
      *success = FALSE;
      return;
   }

   *lincoefx    += bilincoef * refpointy;
   *lincoefy    += bilincoef * refpointx;
   *linconstant += constant;
}

/** computes coefficients of McCormick under- or overestimation of a bilinear term */
void SCIPaddBilinMcCormick(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             bilincoef,          /**< coefficient of bilinear term */
   SCIP_Real             lbx,                /**< lower bound on first variable */
   SCIP_Real             ubx,                /**< upper bound on first variable */
   SCIP_Real             refpointx,          /**< reference point for first variable */
   SCIP_Real             lby,                /**< lower bound on second variable */
   SCIP_Real             uby,                /**< upper bound on second variable */
   SCIP_Real             refpointy,          /**< reference point for second variable */
   SCIP_Bool             overestimate,       /**< whether to compute an overestimator instead of an underestimator */
   SCIP_Real*            lincoefx,           /**< buffer to add coefficient of first  variable in linearization */
   SCIP_Real*            lincoefy,           /**< buffer to add coefficient of second variable in linearization */
   SCIP_Real*            linconstant,        /**< buffer to add constant of linearization */
   SCIP_Bool*            success             /**< buffer to set to FALSE if linearization has failed due to large numbers */
   )
{
   SCIP_Real constant;
   SCIP_Real coefx;
   SCIP_Real coefy;

   assert(scip != NULL);
   assert(!SCIPisInfinity(scip,  lbx));
   assert(!SCIPisInfinity(scip, -ubx));
   assert(!SCIPisInfinity(scip,  lby));
   assert(!SCIPisInfinity(scip, -uby));
   assert(SCIPisInfinity(scip,  -lbx) || SCIPisLE(scip, lbx, ubx));
   assert(SCIPisInfinity(scip,  -lby) || SCIPisLE(scip, lby, uby));
   assert(SCIPisInfinity(scip,  -lbx) || SCIPisLE(scip, lbx, refpointx));
   assert(SCIPisInfinity(scip,  -lby) || SCIPisLE(scip, lby, refpointy));
   assert(SCIPisInfinity(scip,  ubx) || SCIPisGE(scip, ubx, refpointx));
   assert(SCIPisInfinity(scip,  uby) || SCIPisGE(scip, uby, refpointy));
   assert(lincoefx != NULL);
   assert(lincoefy != NULL);
   assert(linconstant != NULL);
   assert(success != NULL);

   if( bilincoef == 0.0 )
      return;

   if( overestimate )
      bilincoef = -bilincoef;

   if( SCIPisRelEQ(scip, lbx, ubx) && SCIPisRelEQ(scip, lby, uby) )
   {
      /* both x and y are mostly fixed */
      SCIP_Real cand1;
      SCIP_Real cand2;
      SCIP_Real cand3;
      SCIP_Real cand4;

      coefx = 0.0;
      coefy = 0.0;

      /* estimate x * y by constant */
      cand1 = lbx * lby;
      cand2 = lbx * uby;
      cand3 = ubx * lby;
      cand4 = ubx * uby;

      /* take most conservative value for underestimator */
      if( bilincoef < 0.0 )
         constant = bilincoef * MAX( MAX(cand1, cand2), MAX(cand3, cand4) );
      else
         constant = bilincoef * MIN( MIN(cand1, cand2), MIN(cand3, cand4) );
   }
   else if( bilincoef > 0.0 )
   {
      /* either x or y is not fixed and coef > 0.0 */
      if( !SCIPisInfinity(scip, -lbx) && !SCIPisInfinity(scip, -lby) &&
         (SCIPisInfinity(scip,  ubx) || SCIPisInfinity(scip,  uby)
            || (uby - refpointy) * (ubx - refpointx) >= (refpointy - lby) * (refpointx - lbx)) )
      {
         if( SCIPisRelEQ(scip, lbx, ubx) )
         {
            /* x*y = lbx * y + (x-lbx) * y >= lbx * y + (x-lbx) * lby >= lbx * y + min{(ubx-lbx) * lby, 0 * lby} */
            coefx    =  0.0;
            coefy    =  bilincoef * lbx;
            constant =  bilincoef * (lby < 0.0 ? (ubx-lbx) * lby : 0.0);
         }
         else if( SCIPisRelEQ(scip, lby, uby) )
         {
            /* x*y = lby * x + (y-lby) * x >= lby * x + (y-lby) * lbx >= lby * x + min{(uby-lby) * lbx, 0 * lbx} */
            coefx    =  bilincoef * lby;
            coefy    =  0.0;
            constant =  bilincoef * (lbx < 0.0 ? (uby-lby) * lbx : 0.0);
         }
         else
         {
            coefx    =  bilincoef * lby;
            coefy    =  bilincoef * lbx;
            constant = -bilincoef * lbx * lby;
         }
      }
      else if( !SCIPisInfinity(scip, ubx) && !SCIPisInfinity(scip, uby) )
      {
         if( SCIPisRelEQ(scip, lbx, ubx) )
         {
            /* x*y = ubx * y + (x-ubx) * y >= ubx * y + (x-ubx) * uby >= ubx * y + min{(lbx-ubx) * uby, 0 * uby} */
            coefx    =  0.0;
            coefy    =  bilincoef * ubx;
            constant =  bilincoef * (uby > 0.0 ? (lbx - ubx) * uby : 0.0);
         }
         else if( SCIPisRelEQ(scip, lby, uby) )
         {
            /* x*y = uby * x + (y-uby) * x >= uby * x + (y-uby) * ubx >= uby * x + min{(lby-uby) * ubx, 0 * ubx} */
            coefx    =  bilincoef * uby;
            coefy    =  0.0;
            constant =  bilincoef * (ubx > 0.0 ? (lby - uby) * ubx : 0.0);
         }
         else
         {
            coefx    =  bilincoef * uby;
            coefy    =  bilincoef * ubx;
            constant = -bilincoef * ubx * uby;
         }
      }
      else
      {
         *success = FALSE;
         return;
      }
   }
   else
   {
      /* either x or y is not fixed and coef < 0.0 */
      if( !SCIPisInfinity(scip,  ubx) && !SCIPisInfinity(scip, -lby) &&
         (SCIPisInfinity(scip, -lbx) || SCIPisInfinity(scip,  uby)
            || (ubx - lbx) * (refpointy - lby) <= (uby - lby) * (refpointx - lbx)) )
      {
         if( SCIPisRelEQ(scip, lbx, ubx) )
         {
            /* x*y = ubx * y + (x-ubx) * y <= ubx * y + (x-ubx) * lby <= ubx * y + max{(lbx-ubx) * lby, 0 * lby} */
            coefx    =  0.0;
            coefy    =  bilincoef * ubx;
            constant =  bilincoef * (lby < 0.0 ? (lbx - ubx) * lby : 0.0);
         }
         else if( SCIPisRelEQ(scip, lby, uby) )
         {
            /* x*y = lby * x + (y-lby) * x <= lby * x + (y-lby) * ubx <= lby * x + max{(uby-lby) * ubx, 0 * ubx} */
            coefx    =  bilincoef * lby;
            coefy    =  0.0;
            constant =  bilincoef * (ubx > 0.0 ? (uby - lby) * ubx : 0.0);
         }
         else
         {
            coefx    =  bilincoef * lby;
            coefy    =  bilincoef * ubx;
            constant = -bilincoef * ubx * lby;
         }
      }
      else if( !SCIPisInfinity(scip, -lbx) && !SCIPisInfinity(scip, uby) )
      {
         if( SCIPisRelEQ(scip, lbx, ubx) )
         {
            /* x*y = lbx * y + (x-lbx) * y <= lbx * y + (x-lbx) * uby <= lbx * y + max{(ubx-lbx) * uby, 0 * uby} */
            coefx    =  0.0;
            coefy    =  bilincoef * lbx;
            constant =  bilincoef * (uby > 0.0 ? (ubx - lbx) * uby : 0.0);
         }
         else if( SCIPisRelEQ(scip, lby, uby) )
         {
            /* x*y = uby * x + (y-uby) * x <= uby * x + (y-uby) * lbx <= uby * x + max{(lby-uby) * lbx, 0 * lbx} */
            coefx    =  bilincoef * uby;
            coefy    =  0.0;
            constant =  bilincoef * (lbx < 0.0 ? (lby - uby) * lbx : 0.0);
         }
         else
         {
            coefx    =  bilincoef * uby;
            coefy    =  bilincoef * lbx;
            constant = -bilincoef * lbx * uby;
         }
      }
      else
      {
         *success = FALSE;
         return;
      }
   }

   if( SCIPisInfinity(scip, REALABS(coefx)) || SCIPisInfinity(scip, REALABS(coefy))
      || SCIPisInfinity(scip, REALABS(constant)) )
   {
      *success = FALSE;
      return;
   }

   if( overestimate )
   {
      coefx    = -coefx;
      coefy    = -coefy;
      constant = -constant;
   }

   SCIPdebugMsg(scip, "%.15g * x[%.15g,%.15g] * y[%.15g,%.15g] %c= %.15g * x %+.15g * y %+.15g\n", bilincoef, lbx, ubx,
      lby, uby, overestimate ? '<' : '>', coefx, coefy, constant);

   *lincoefx    += coefx;
   *lincoefy    += coefy;
   *linconstant += constant;
}


/** computes coefficients of linearization of a bilinear term in a reference point when given a linear inequality
 *  involving only the variables of the bilinear term
 *
 *  @note the formulas are extracted from "Convex envelopes of bivariate functions through the solution of KKT systems"
 *        by Marco Locatelli
 */
void SCIPcomputeBilinEnvelope1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             bilincoef,          /**< coefficient of bilinear term */
   SCIP_Real             lbx,                /**< lower bound on first variable */
   SCIP_Real             ubx,                /**< upper bound on first variable */
   SCIP_Real             refpointx,          /**< reference point for first variable */
   SCIP_Real             lby,                /**< lower bound on second variable */
   SCIP_Real             uby,                /**< upper bound on second variable */
   SCIP_Real             refpointy,          /**< reference point for second variable */
   SCIP_Bool             overestimate,       /**< whether to compute an overestimator instead of an underestimator */
   SCIP_Real             xcoef,              /**< x coefficient of linear inequality; must be in {-1,0,1} */
   SCIP_Real             ycoef,              /**< y coefficient of linear inequality */
   SCIP_Real             constant,           /**< constant of linear inequality */
   SCIP_Real* RESTRICT   lincoefx,           /**< buffer to store coefficient of first  variable in linearization */
   SCIP_Real* RESTRICT   lincoefy,           /**< buffer to store coefficient of second variable in linearization */
   SCIP_Real* RESTRICT   linconstant,        /**< buffer to store constant of linearization */
   SCIP_Bool* RESTRICT   success             /**< buffer to store whether linearization was successful */
   )
{
   SCIP_Real xs[2] = {lbx, ubx};
   SCIP_Real ys[2] = {lby, uby};
   SCIP_Real minx;
   SCIP_Real maxx;
   SCIP_Real miny;
   SCIP_Real maxy;
   SCIP_Real QUAD(lincoefyq);
   SCIP_Real QUAD(lincoefxq);
   SCIP_Real QUAD(linconstantq);
   SCIP_Real QUAD(denomq);
   SCIP_Real QUAD(mjq);
   SCIP_Real QUAD(qjq);
   SCIP_Real QUAD(xjq);
   SCIP_Real QUAD(yjq);
   SCIP_Real QUAD(tmpq);
   SCIP_Real vx;
   SCIP_Real vy;
   int n;
   int i;

   assert(scip != NULL);
   assert(!SCIPisInfinity(scip,  lbx));
   assert(!SCIPisInfinity(scip, -ubx));
   assert(!SCIPisInfinity(scip,  lby));
   assert(!SCIPisInfinity(scip, -uby));
   assert(SCIPisLE(scip, lbx, ubx));
   assert(SCIPisLE(scip, lby, uby));
   assert(SCIPisLE(scip, lbx, refpointx));
   assert(SCIPisGE(scip, ubx, refpointx));
   assert(SCIPisLE(scip, lby, refpointy));
   assert(SCIPisGE(scip, uby, refpointy));
   assert(lincoefx != NULL);
   assert(lincoefy != NULL);
   assert(linconstant != NULL);
   assert(success != NULL);
   assert(xcoef == 0.0 || xcoef == -1.0 || xcoef == 1.0); /*lint !e777*/
   assert(ycoef != SCIP_INVALID && ycoef != 0.0); /*lint !e777*/
   assert(constant != SCIP_INVALID); /*lint !e777*/

   *success = FALSE;
   *lincoefx = SCIP_INVALID;
   *lincoefy = SCIP_INVALID;
   *linconstant = SCIP_INVALID;

   /* reference point does not satisfy linear inequality */
   if( SCIPisFeasGT(scip, xcoef * refpointx - ycoef * refpointy - constant, 0.0) )
      return;

   /* compute minimal and maximal bounds on x and y for accepting the reference point */
   minx = lbx + 0.01 * (ubx-lbx);
   maxx = ubx - 0.01 * (ubx-lbx);
   miny = lby + 0.01 * (uby-lby);
   maxy = uby - 0.01 * (uby-lby);

   /* check whether the reference point is in [minx,maxx]x[miny,maxy] */
   if( SCIPisLE(scip, refpointx, minx) || SCIPisGE(scip, refpointx, maxx)
      || SCIPisLE(scip, refpointy, miny) || SCIPisGE(scip, refpointy, maxy) )
      return;

   /* always consider xy without the bilinear coefficient */
   if( bilincoef < 0.0 )
      overestimate = !overestimate;

   /* we use same notation as in "Convex envelopes of bivariate functions through the solution of KKT systems", 2016 */
   /* mj = xcoef / ycoef */
   SCIPquadprecDivDD(mjq, xcoef, ycoef);

   /* qj = -constant / ycoef */
   SCIPquadprecDivDD(qjq, -constant, ycoef);

   /* mj > 0 => underestimate; mj < 0 => overestimate */
   if( SCIPisNegative(scip, QUAD_TO_DBL(mjq)) != overestimate )
      return;

   /* get the corner point that satisfies the linear inequality xcoef*x <= ycoef*y + constant */
   if( !overestimate )
   {
      ys[0] = uby;
      ys[1] = lby;
   }

   vx = SCIP_INVALID;
   vy = SCIP_INVALID;
   n = 0;
   for( i = 0; i < 2; ++i )
   {
      SCIP_Real activity = xcoef * xs[i] - ycoef * ys[i] - constant;
      if( SCIPisLE(scip, activity, 0.0) )
      {
         /* corner point is satisfies inequality */
         vx = xs[i];
         vy = ys[i];
      }
      else if( SCIPisFeasGT(scip, activity, 0.0) )
         /* corner point is clearly cut off */
         ++n;
   }

   /* skip if no corner point satisfies the inequality or if no corner point is cut off (that is, all corner points satisfy the inequality almost [1e-9..1e-6]) */
   if( n != 1 || vx == SCIP_INVALID || vy == SCIP_INVALID ) /*lint !e777*/
      return;

   /* denom = mj*(refpointx - vx) + vy - refpointy */
   SCIPquadprecSumDD(denomq, refpointx, -vx); /* refpoint - vx */
   SCIPquadprecProdQQ(denomq, denomq, mjq); /* mj * (refpoint - vx) */
   SCIPquadprecSumQD(denomq, denomq, vy); /* mj * (refpoint - vx) + vy */
   SCIPquadprecSumQD(denomq, denomq, -refpointy); /* mj * (refpoint - vx) + vy - refpointy */

   if( SCIPisZero(scip, QUAD_TO_DBL(denomq)) )
      return;

   /* (xj,yj) is the projection onto the line xcoef*x = ycoef*y + constant */
   /* xj = (refpointx*(vy - qj) - vx*(refpointy - qj)) / denom */
   SCIPquadprecProdQD(xjq, qjq, -1.0); /* - qj */
   SCIPquadprecSumQD(xjq, xjq, vy); /* vy - qj */
   SCIPquadprecProdQD(xjq, xjq, refpointx); /* refpointx * (vy - qj) */
   SCIPquadprecProdQD(tmpq, qjq, -1.0); /* - qj */
   SCIPquadprecSumQD(tmpq, tmpq, refpointy); /* refpointy - qj */
   SCIPquadprecProdQD(tmpq, tmpq, -vx); /* - vx * (refpointy - qj) */
   SCIPquadprecSumQQ(xjq, xjq, tmpq); /* refpointx * (vy - qj) - vx * (refpointy - qj) */
   SCIPquadprecDivQQ(xjq, xjq, denomq); /* (refpointx * (vy - qj) - vx * (refpointy - qj)) / denom */

   /* yj = mj * xj + qj */
   SCIPquadprecProdQQ(yjq, mjq, xjq);
   SCIPquadprecSumQQ(yjq, yjq, qjq);

   assert(SCIPisFeasEQ(scip, xcoef*QUAD_TO_DBL(xjq) - ycoef*QUAD_TO_DBL(yjq) - constant, 0.0));

   /* check whether the projection is in [minx,maxx] x [miny,maxy]; this avoids numerical difficulties when the
    * projection is close to the variable bounds
    */
   if( SCIPisLE(scip, QUAD_TO_DBL(xjq), minx) || SCIPisGE(scip, QUAD_TO_DBL(xjq), maxx)
      || SCIPisLE(scip, QUAD_TO_DBL(yjq), miny) || SCIPisGE(scip, QUAD_TO_DBL(yjq), maxy) )
      return;

   assert(vy - QUAD_TO_DBL(mjq)*vx - QUAD_TO_DBL(qjq) != 0.0);

   /* lincoefy = (mj*SQR(xj) - 2.0*mj*vx*xj - qj*vx + vx*vy) / (vy - mj*vx - qj) */
   SCIPquadprecSquareQ(lincoefyq, xjq); /* xj^2 */
   SCIPquadprecProdQQ(lincoefyq, lincoefyq, mjq); /* mj * xj^2 */
   SCIPquadprecProdQQ(tmpq, mjq, xjq); /* mj * xj */
   SCIPquadprecProdQD(tmpq, tmpq, -2.0 * vx); /* -2 * vx * mj * xj */
   SCIPquadprecSumQQ(lincoefyq, lincoefyq, tmpq); /* mj * xj^2 -2 * vx * mj * xj */
   SCIPquadprecProdQD(tmpq, qjq, -vx); /* -qj * vx */
   SCIPquadprecSumQQ(lincoefyq, lincoefyq, tmpq); /* mj * xj^2 -2 * vx * mj * xj -qj * vx */
   SCIPquadprecProdDD(tmpq, vx, vy); /* vx * vy */
   SCIPquadprecSumQQ(lincoefyq, lincoefyq, tmpq); /* mj * xj^2 -2 * vx * mj * xj -qj * vx + vx * vy */
   SCIPquadprecProdQD(tmpq, mjq, vx); /* mj * vx */
   SCIPquadprecSumQD(tmpq, tmpq, -vy); /* -vy + mj * vx */
   SCIPquadprecSumQQ(tmpq, tmpq, qjq); /* -vy + mj * vx + qj */
   QUAD_SCALE(tmpq, -1.0); /* vy - mj * vx - qj */
   SCIPquadprecDivQQ(lincoefyq, lincoefyq, tmpq); /* (mj * xj^2 -2 * vx * mj * xj -qj * vx + vx * vy) / (vy - mj * vx - qj) */

   /* lincoefx = 2.0*mj*xj + qj - mj*(*lincoefy) */
   SCIPquadprecProdQQ(lincoefxq, mjq, xjq); /* mj * xj */
   QUAD_SCALE(lincoefxq, 2.0); /* 2 * mj * xj */
   SCIPquadprecSumQQ(lincoefxq, lincoefxq, qjq); /* 2 * mj * xj + qj */
   SCIPquadprecProdQQ(tmpq, mjq, lincoefyq); /* mj * lincoefy */
   QUAD_SCALE(tmpq, -1.0); /* - mj * lincoefy */
   SCIPquadprecSumQQ(lincoefxq, lincoefxq, tmpq); /* 2 * mj * xj + qj - mj * lincoefy */

   /* linconstant = -mj*SQR(xj) - (*lincoefy)*qj */
   SCIPquadprecSquareQ(linconstantq, xjq); /* xj^2 */
   SCIPquadprecProdQQ(linconstantq, linconstantq, mjq); /* mj * xj^2 */
   QUAD_SCALE(linconstantq, -1.0); /* - mj * xj^2 */
   SCIPquadprecProdQQ(tmpq, lincoefyq, qjq); /* lincoefy * qj */
   QUAD_SCALE(tmpq, -1.0); /* - lincoefy * qj */
   SCIPquadprecSumQQ(linconstantq, linconstantq, tmpq); /* - mj * xj^2 - lincoefy * qj */

   /* consider the bilinear coefficient */
   SCIPquadprecProdQD(lincoefxq, lincoefxq, bilincoef);
   SCIPquadprecProdQD(lincoefyq, lincoefyq, bilincoef);
   SCIPquadprecProdQD(linconstantq, linconstantq, bilincoef);
   *lincoefx = QUAD_TO_DBL(lincoefxq);
   *lincoefy = QUAD_TO_DBL(lincoefyq);
   *linconstant = QUAD_TO_DBL(linconstantq);

   /* cut needs to be tight at (vx,vy) and (xj,yj); otherwise we consider the cut to be numerically bad */
   *success = SCIPisFeasEQ(scip, (*lincoefx)*vx + (*lincoefy)*vy + (*linconstant), bilincoef*vx*vy)
      && SCIPisFeasEQ(scip, (*lincoefx)*QUAD_TO_DBL(xjq) + (*lincoefy)*QUAD_TO_DBL(yjq) + (*linconstant), bilincoef*QUAD_TO_DBL(xjq)*QUAD_TO_DBL(yjq));

#ifndef NDEBUG
   {
      SCIP_Real activity = (*lincoefx)*refpointx + (*lincoefy)*refpointy + (*linconstant);

      /* cut needs to under- or overestimate the bilinear term at the reference point */
      if( bilincoef < 0.0 )
         overestimate = !overestimate;

      if( overestimate )
         assert(SCIPisFeasGE(scip, activity, bilincoef*refpointx*refpointy));
      else
         assert(SCIPisFeasLE(scip, activity, bilincoef*refpointx*refpointy));
   }
#endif
}

/** helper function to compute the convex envelope of a bilinear term when two linear inequalities are given; we
 *  use the same notation and formulas as in Locatelli 2016
 */
static
void computeBilinEnvelope2(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             x,                  /**< reference point for x */
   SCIP_Real             y,                  /**< reference point for y */
   SCIP_Real             mi,                 /**< coefficient of x in the first linear inequality */
   SCIP_Real             qi,                 /**< constant in the first linear inequality */
   SCIP_Real             mj,                 /**< coefficient of x in the second linear inequality */
   SCIP_Real             qj,                 /**< constant in the second linear inequality */
   SCIP_Real* RESTRICT   xi,                 /**< buffer to store x coordinate of the first point */
   SCIP_Real* RESTRICT   yi,                 /**< buffer to store y coordinate of the first point */
   SCIP_Real* RESTRICT   xj,                 /**< buffer to store x coordinate of the second point */
   SCIP_Real* RESTRICT   yj,                 /**< buffer to store y coordinate of the second point */
   SCIP_Real* RESTRICT   xcoef,              /**< buffer to store the x coefficient of the envelope */
   SCIP_Real* RESTRICT   ycoef,              /**< buffer to store the y coefficient of the envelope */
   SCIP_Real* RESTRICT   constant            /**< buffer to store the constant of the envelope */
   )
{
   SCIP_Real QUAD(xiq);
   SCIP_Real QUAD(yiq);
   SCIP_Real QUAD(xjq);
   SCIP_Real QUAD(yjq);
   SCIP_Real QUAD(xcoefq);
   SCIP_Real QUAD(ycoefq);
   SCIP_Real QUAD(constantq);
   SCIP_Real QUAD(tmpq);

   assert(xi != NULL);
   assert(yi != NULL);
   assert(xj != NULL);
   assert(yj != NULL);
   assert(xcoef != NULL);
   assert(ycoef != NULL);
   assert(constant != NULL);

   if( SCIPisEQ(scip, mi, mj) )
   {
      /* xi = (x + mi * y - qi) / (2.0*mi) */
      SCIPquadprecProdDD(xiq, mi, y);
      SCIPquadprecSumQD(xiq, xiq, x);
      SCIPquadprecSumQD(xiq, xiq, -qi);
      SCIPquadprecDivQD(xiq, xiq, 2.0 * mi);
      assert(EPSEQ((x + mi * y - qi) / (2.0*mi), QUAD_TO_DBL(xiq), 1e-3));

      /* yi = mi*(*xi) + qi */
      SCIPquadprecProdQD(yiq, xiq, mi);
      SCIPquadprecSumQD(yiq, yiq, qi);
      assert(EPSEQ(mi*QUAD_TO_DBL(xiq) + qi, QUAD_TO_DBL(yiq), 1e-3));

      /* xj = (*xi) + (qi - qj)/ (2.0*mi) */
      SCIPquadprecSumDD(xjq, qi, -qj);
      SCIPquadprecDivQD(xjq, xjq, 2.0 * mi);
      SCIPquadprecSumQQ(xjq, xjq, xiq);
      assert(EPSEQ(QUAD_TO_DBL(xiq) + (qi - qj)/ (2.0*mi), QUAD_TO_DBL(xjq), 1e-3));

      /* yj = mj * (*xj) + qj */
      SCIPquadprecProdQD(yjq, xjq, mj);
      SCIPquadprecSumQD(yjq, yjq, qj);
      assert(EPSEQ(mj * QUAD_TO_DBL(xjq) + qj, QUAD_TO_DBL(yjq), 1e-3));

      /* ycoef = (*xi) + (qi - qj) / (4.0*mi) note that this is wrong in Locatelli 2016 */
      SCIPquadprecSumDD(ycoefq, qi, -qj);
      SCIPquadprecDivQD(ycoefq, ycoefq, 4.0 * mi);
      SCIPquadprecSumQQ(ycoefq, ycoefq, xiq);
      assert(EPSEQ(QUAD_TO_DBL(xiq) + (qi - qj) / (4.0*mi), QUAD_TO_DBL(ycoefq), 1e-3));

      /* xcoef = 2.0*mi*(*xi) - mi * (*ycoef) + qi */
      SCIPquadprecProdQD(xcoefq, xiq, 2.0 * mi);
      SCIPquadprecProdQD(tmpq, ycoefq, -mi);
      SCIPquadprecSumQQ(xcoefq, xcoefq, tmpq);
      SCIPquadprecSumQD(xcoefq, xcoefq, qi);
      assert(EPSEQ(2.0*mi*QUAD_TO_DBL(xiq) - mi * QUAD_TO_DBL(ycoefq) + qi, QUAD_TO_DBL(xcoefq), 1e-3));

      /* constant = -mj*SQR(*xj) - (*ycoef) * qj */
      SCIPquadprecSquareQ(constantq, xjq);
      SCIPquadprecProdQD(constantq, constantq, -mj);
      SCIPquadprecProdQD(tmpq, ycoefq, -qj);
      SCIPquadprecSumQQ(constantq, constantq, tmpq);
      /* assert(EPSEQ(-mj*SQR(QUAD_TO_DBL(xjq)) - QUAD_TO_DBL(ycoefq) * qj, QUAD_TO_DBL(constantq), 1e-3)); */

      *xi = QUAD_TO_DBL(xiq);
      *yi = QUAD_TO_DBL(yiq);
      *xj = QUAD_TO_DBL(xjq);
      *yj = QUAD_TO_DBL(yjq);
      *ycoef = QUAD_TO_DBL(ycoefq);
      *xcoef = QUAD_TO_DBL(xcoefq);
      *constant = QUAD_TO_DBL(constantq);
   }
   else if( mi > 0.0 )
   {
      assert(mj > 0.0);

      /* xi = (y + SQRT(mi*mj)*x - qi) / (REALABS(mi) + SQRT(mi*mj)) */
      SCIPquadprecProdDD(xiq, mi, mj);
      SCIPquadprecSqrtQ(xiq, xiq);
      SCIPquadprecProdQD(xiq, xiq, x);
      SCIPquadprecSumQD(xiq, xiq, y);
      SCIPquadprecSumQD(xiq, xiq, -qi); /* (y + SQRT(mi*mj)*x - qi) */
      SCIPquadprecProdDD(tmpq, mi, mj);
      SCIPquadprecSqrtQ(tmpq, tmpq);
      SCIPquadprecSumQD(tmpq, tmpq, REALABS(mi)); /* REALABS(mi) + SQRT(mi*mj) */
      SCIPquadprecDivQQ(xiq, xiq, tmpq);
      assert(EPSEQ((y + SQRT(mi*mj)*x - qi) / (REALABS(mi) + SQRT(mi*mj)), QUAD_TO_DBL(xiq), 1e-3));

      /* yi = mi*(*xi) + qi */
      SCIPquadprecProdQD(yiq, xiq, mi);
      SCIPquadprecSumQD(yiq, yiq, qi);
      assert(EPSEQ(mi*(QUAD_TO_DBL(xiq)) + qi, QUAD_TO_DBL(yiq), 1e-3));

      /* xj = (y + SQRT(mi*mj)*x - qj) / (REALABS(mj) + SQRT(mi*mj)) */
      SCIPquadprecProdDD(xjq, mi, mj);
      SCIPquadprecSqrtQ(xjq, xjq);
      SCIPquadprecProdQD(xjq, xjq, x);
      SCIPquadprecSumQD(xjq, xjq, y);
      SCIPquadprecSumQD(xjq, xjq, -qj); /* (y + SQRT(mi*mj)*x - qj) */
      SCIPquadprecProdDD(tmpq, mi, mj);
      SCIPquadprecSqrtQ(tmpq, tmpq);
      SCIPquadprecSumQD(tmpq, tmpq, REALABS(mj)); /* REALABS(mj) + SQRT(mi*mj) */
      SCIPquadprecDivQQ(xjq, xjq, tmpq);
      assert(EPSEQ((y + SQRT(mi*mj)*x - qj) / (REALABS(mj) + SQRT(mi*mj)), QUAD_TO_DBL(xjq), 1e-3));

      /* yj = mj*(*xj) + qj */
      SCIPquadprecProdQD(yjq, xjq, mj);
      SCIPquadprecSumQD(yjq, yjq, qj);
      assert(EPSEQ(mj*QUAD_TO_DBL(xjq) + qj, QUAD_TO_DBL(yjq), 1e-3));

      /* ycoef = (2.0*mj*(*xj) + qj - 2.0*mi*(*xi) - qi) / (mj - mi) */
      SCIPquadprecProdQD(ycoefq, xjq, 2.0 * mj);
      SCIPquadprecSumQD(ycoefq, ycoefq, qj);
      SCIPquadprecProdQD(tmpq, xiq, -2.0 * mi);
      SCIPquadprecSumQQ(ycoefq, ycoefq, tmpq);
      SCIPquadprecSumQD(ycoefq, ycoefq, -qi);
      SCIPquadprecSumDD(tmpq, mj, -mi);
      SCIPquadprecDivQQ(ycoefq, ycoefq, tmpq);
      assert(EPSEQ((2.0*mj*QUAD_TO_DBL(xjq) + qj - 2.0*mi*QUAD_TO_DBL(xiq) - qi) / (mj - mi), QUAD_TO_DBL(ycoefq), 1e-3));

      /* xcoef = 2.0*mj*(*xj) + qj - mj*(*ycoef) */
      SCIPquadprecProdQD(xcoefq, xjq, 2.0 * mj);
      SCIPquadprecSumQD(xcoefq, xcoefq, qj);
      SCIPquadprecProdQD(tmpq, ycoefq, -mj);
      SCIPquadprecSumQQ(xcoefq, xcoefq, tmpq);
      assert(EPSEQ(2.0*mj*QUAD_TO_DBL(xjq) + qj - mj*QUAD_TO_DBL(ycoefq), QUAD_TO_DBL(xcoefq), 1e-3));

      /* constant = -mj*SQR(*xj) - (*ycoef) * qj */
      SCIPquadprecSquareQ(constantq, xjq);
      SCIPquadprecProdQD(constantq, constantq, -mj);
      SCIPquadprecProdQD(tmpq, ycoefq, -qj);
      SCIPquadprecSumQQ(constantq, constantq, tmpq);
      /* assert(EPSEQ(-mj*SQR(QUAD_TO_DBL(xjq)) - QUAD_TO_DBL(ycoefq) * qj, QUAD_TO_DBL(constantq), 1e-3)); */

      *xi = QUAD_TO_DBL(xiq);
      *yi = QUAD_TO_DBL(yiq);
      *xj = QUAD_TO_DBL(xjq);
      *yj = QUAD_TO_DBL(yjq);
      *ycoef = QUAD_TO_DBL(ycoefq);
      *xcoef = QUAD_TO_DBL(xcoefq);
      *constant = QUAD_TO_DBL(constantq);
   }
   else
   {
      assert(mi < 0.0 && mj < 0.0);

      /* apply variable transformation x = -x in case for overestimation */
      computeBilinEnvelope2(scip, -x, y, -mi, qi, -mj, qj, xi, yi, xj, yj, xcoef, ycoef, constant);

      /* revert transformation; multiply cut by -1 and change -x by x */
      *xi = -(*xi);
      *xj = -(*xj);
      *ycoef = -(*ycoef);
      *constant = -(*constant);
   }
}

/** computes coefficients of linearization of a bilinear term in a reference point when given two linear inequality
 *  involving only the variables of the bilinear term
 *
 *  @note the formulas are extracted from "Convex envelopes of bivariate functions through the solution of KKT systems"
 *        by Marco Locatelli
 *
 */
void SCIPcomputeBilinEnvelope2(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             bilincoef,          /**< coefficient of bilinear term */
   SCIP_Real             lbx,                /**< lower bound on first variable */
   SCIP_Real             ubx,                /**< upper bound on first variable */
   SCIP_Real             refpointx,          /**< reference point for first variable */
   SCIP_Real             lby,                /**< lower bound on second variable */
   SCIP_Real             uby,                /**< upper bound on second variable */
   SCIP_Real             refpointy,          /**< reference point for second variable */
   SCIP_Bool             overestimate,       /**< whether to compute an overestimator instead of an underestimator */
   SCIP_Real             xcoef1,             /**< x coefficient of linear inequality; must be in {-1,0,1} */
   SCIP_Real             ycoef1,             /**< y coefficient of linear inequality */
   SCIP_Real             constant1,          /**< constant of linear inequality */
   SCIP_Real             xcoef2,             /**< x coefficient of linear inequality; must be in {-1,0,1} */
   SCIP_Real             ycoef2,             /**< y coefficient of linear inequality */
   SCIP_Real             constant2,          /**< constant of linear inequality */
   SCIP_Real* RESTRICT   lincoefx,           /**< buffer to store coefficient of first  variable in linearization */
   SCIP_Real* RESTRICT   lincoefy,           /**< buffer to store coefficient of second variable in linearization */
   SCIP_Real* RESTRICT   linconstant,        /**< buffer to store constant of linearization */
   SCIP_Bool* RESTRICT   success             /**< buffer to store whether linearization was successful */
   )
{
   SCIP_Real mi, mj, qi, qj, xi, xj, yi, yj;
   SCIP_Real xcoef, ycoef, constant;
   SCIP_Real minx, maxx, miny, maxy;

   assert(scip != NULL);
   assert(!SCIPisInfinity(scip,  lbx));
   assert(!SCIPisInfinity(scip, -ubx));
   assert(!SCIPisInfinity(scip,  lby));
   assert(!SCIPisInfinity(scip, -uby));
   assert(SCIPisLE(scip, lbx, ubx));
   assert(SCIPisLE(scip, lby, uby));
   assert(SCIPisLE(scip, lbx, refpointx));
   assert(SCIPisGE(scip, ubx, refpointx));
   assert(SCIPisLE(scip, lby, refpointy));
   assert(SCIPisGE(scip, uby, refpointy));
   assert(lincoefx != NULL);
   assert(lincoefy != NULL);
   assert(linconstant != NULL);
   assert(success != NULL);
   assert(xcoef1 != 0.0 && xcoef1 != SCIP_INVALID); /*lint !e777*/
   assert(ycoef1 != SCIP_INVALID && ycoef1 != 0.0); /*lint !e777*/
   assert(constant1 != SCIP_INVALID); /*lint !e777*/
   assert(xcoef2 != 0.0 && xcoef2 != SCIP_INVALID); /*lint !e777*/
   assert(ycoef2 != SCIP_INVALID && ycoef2 != 0.0); /*lint !e777*/
   assert(constant2 != SCIP_INVALID); /*lint !e777*/

   *success = FALSE;
   *lincoefx = SCIP_INVALID;
   *lincoefy = SCIP_INVALID;
   *linconstant = SCIP_INVALID;

   /* reference point does not satisfy linear inequalities */
   if( SCIPisFeasGT(scip, xcoef1 * refpointx - ycoef1 * refpointy - constant1, 0.0)
      || SCIPisFeasGT(scip, xcoef2 * refpointx - ycoef2 * refpointy - constant2, 0.0) )
      return;

   /* compute minimal and maximal bounds on x and y for accepting the reference point */
   minx = lbx + 0.01 * (ubx-lbx);
   maxx = ubx - 0.01 * (ubx-lbx);
   miny = lby + 0.01 * (uby-lby);
   maxy = uby - 0.01 * (uby-lby);

   /* check the reference point is in the interior of the domain */
   if( SCIPisLE(scip, refpointx, minx) || SCIPisGE(scip, refpointx, maxx)
      || SCIPisLE(scip, refpointy, miny) || SCIPisFeasGE(scip, refpointy, maxy) )
      return;

   /* the sign of the x-coefficients of the two inequalities must be different; otherwise the convex or concave
    * envelope can be computed via SCIPcomputeBilinEnvelope1 for each inequality separately
    */
   if( (xcoef1 > 0) == (xcoef2 > 0) )
      return;

   /* always consider xy without the bilinear coefficient */
   if( bilincoef < 0.0 )
      overestimate = !overestimate;

   /* we use same notation as in "Convex envelopes of bivariate functions through the solution of KKT systems", 2016 */
   mi = xcoef1 / ycoef1;
   qi = -constant1 / ycoef1;
   mj = xcoef2 / ycoef2;
   qj = -constant2 / ycoef2;

   /* mi, mj > 0 => underestimate; mi, mj < 0 => overestimate */
   if( SCIPisNegative(scip, mi) != overestimate || SCIPisNegative(scip, mj) != overestimate )
      return;

   /* compute cut according to Locatelli 2016 */
   computeBilinEnvelope2(scip, refpointx, refpointy, mi, qi, mj, qj, &xi, &yi, &xj, &yj, &xcoef, &ycoef, &constant);
   assert(SCIPisRelEQ(scip, mi*xi + qi, yi));
   assert(SCIPisRelEQ(scip, mj*xj + qj, yj));

   /* it might happen that (xi,yi) = (xj,yj) if the two lines intersect */
   if( SCIPisEQ(scip, xi, xj) && SCIPisEQ(scip, yi, yj) )
      return;

   /* check whether projected points are in the interior */
   if( SCIPisLE(scip, xi, minx) || SCIPisGE(scip, xi, maxx) || SCIPisLE(scip, yi, miny) || SCIPisGE(scip, yi, maxy) )
      return;
   if( SCIPisLE(scip, xj, minx) || SCIPisGE(scip, xj, maxx) || SCIPisLE(scip, yj, miny) || SCIPisGE(scip, yj, maxy) )
      return;

   *lincoefx = bilincoef * xcoef;
   *lincoefy = bilincoef * ycoef;
   *linconstant = bilincoef * constant;

   /* cut needs to be tight at (vx,vy) and (xj,yj) */
   *success = SCIPisFeasEQ(scip, (*lincoefx)*xi + (*lincoefy)*yi + (*linconstant), bilincoef*xi*yi)
      && SCIPisFeasEQ(scip, (*lincoefx)*xj + (*lincoefy)*yj + (*linconstant), bilincoef*xj*yj);

#ifndef NDEBUG
   {
      SCIP_Real activity = (*lincoefx)*refpointx + (*lincoefy)*refpointy + (*linconstant);

      /* cut needs to under- or overestimate the bilinear term at the reference point */
      if( bilincoef < 0.0 )
         overestimate = !overestimate;

      if( overestimate )
         assert(SCIPisFeasGE(scip, activity, bilincoef*refpointx*refpointy));
      else
         assert(SCIPisFeasLE(scip, activity, bilincoef*refpointx*refpointy));
   }
#endif
}
