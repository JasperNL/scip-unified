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

/**@file   presol_dualsparsify.c
 * @brief  cancel non-zeros of the constraint matrix
 * @author Dieter Weninger
 * @author Robert Lion Gottwald
 * @author Ambros Gleixner
 *
 * This presolver attempts to cancel non-zero entries of the constraint
 * matrix by adding scaled variables to other variables.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "blockmemshell/memory.h"
#include "scip/cons_linear.h"
#include "scip/presol_dualsparsify.h"
#include "scip/pub_cons.h"
#include "scip/pub_matrix.h"
#include "scip/pub_message.h"
#include "scip/pub_misc.h"
#include "scip/pub_misc_sort.h"
#include "scip/pub_presol.h"
#include "scip/pub_var.h"
#include "scip/scip_cons.h"
#include "scip/scip_general.h"
#include "scip/scip_mem.h"
#include "scip/scip_message.h"
#include "scip/scip_nlp.h"
#include "scip/scip_numerics.h"
#include "scip/scip_param.h"
#include "scip/scip_presol.h"
#include "scip/scip_pricer.h"
#include "scip/scip_prob.h"
#include "scip/scip_probing.h"
#include "scip/scip_timing.h"
#include "scip/scip_var.h"
#include "scip/cons_varbound.h"
#include <string.h>

#define PRESOL_NAME            "dualsparsify"
#define PRESOL_DESC            "eliminate non-zero coefficients"

#define PRESOL_PRIORITY            -24000    /**< priority of the presolver (>= 0: before, < 0: after constraint handlers) */
#define PRESOL_MAXROUNDS               -1    /**< maximal number of presolving rounds the presolver participates in (-1: no limit) */
#define PRESOL_TIMING           SCIP_PRESOLTIMING_EXHAUSTIVE /* timing of the presolver (fast, medium, or exhaustive) */

#define DEFAULT_ENABLECOPY           TRUE    /**< should dualsparsify presolver be copied to sub-SCIPs? */
#define DEFAULT_CANCELLINEAR         TRUE    /**< should we cancel nonzeros in constraints of the linear constraint handler? */
#define DEFAULT_PRESERVEINTCOEFS     TRUE    /**< should we forbid cancellations that destroy integer coefficients? */
#define DEFAULT_MAX_CONT_FILLIN         0    /**< default value for the maximal fillin for continuous variables */
#define DEFAULT_MAX_BIN_FILLIN          0    /**< default value for the maximal fillin for binary variables */
#define DEFAULT_MAX_INT_FILLIN          0    /**< default value for the maximal fillin for integer variables (including binary) */
#define DEFAULT_MAXNONZEROS            -1    /**< maximal support of one equality to be used for cancelling (-1: no limit) */
#define DEFAULT_MAXCONSIDEREDNONZEROS  70    /**< maximal number of considered non-zeros within one row (-1: no limit) */
#define DEFAULT_ROWSORT               'd'    /**< order in which to process inequalities ('n'o sorting, 'i'ncreasing nonzeros, 'd'ecreasing nonzeros) */
#define DEFAULT_MAXRETRIEVEFAC      100.0    /**< limit on the number of useless vs. useful hashtable retrieves as a multiple of the number of constraints */
#define DEFAULT_WAITINGFAC            2.0    /**< number of calls to wait until next execution as a multiple of the number of useless calls */

#define MAXSCALE                   1000.0    /**< maximal allowed scale for cancelling non-zeros */


/*
 * Data structures
 */

/** presolver data */
struct SCIP_PresolData
{
   int                   ncancels;           /**< total number of canceled nonzeros (net value, i.e., removed minus added nonzeros) */
   int                   nfillin;            /**< total number of added nonzeros */
   int                   nfailures;          /**< number of calls to presolver without success */
   int                   nwaitingcalls;      /**< number of presolver calls until next real execution */
   int                   maxcontfillin;      /**< maximal fillin for continuous variables */
   int                   maxintfillin;       /**< maximal fillin for integer variables*/
   int                   maxbinfillin;       /**< maximal fillin for binary variables */
   int                   maxnonzeros;        /**< maximal support of one equality to be used for cancelling (-1: no limit) */
   int                   maxconsiderednonzeros;/**< maximal number of considered non-zeros within one row (-1: no limit) */
   SCIP_Real             maxretrievefac;     /**< limit on the number of useless vs. useful hashtable retrieves as a multiple of the number of constraints */
   SCIP_Real             waitingfac;         /**< number of calls to wait until next execution as a multiple of the number of useless calls */
   char                  rowsort;            /**< order in which to process inequalities ('n'o sorting, 'i'ncreasing nonzeros, 'd'ecreasing nonzeros) */
   SCIP_Bool             enablecopy;         /**< should dualsparsify presolver be copied to sub-SCIPs? */
   SCIP_Bool             cancellinear;       /**< should we cancel nonzeros in constraints of the linear constraint handler? */
   SCIP_Bool             preserveintcoefs;   /**< should we forbid cancellations that destroy integer coefficients? */
};

/** structure representing a pair of variables in a row; used for lookup in a hashtable */
struct RowVarPair
{
   int rowindex;
   int varindex1;
   int varindex2;
   SCIP_Real varcoef1;
   SCIP_Real varcoef2;
};

typedef struct RowVarPair ROWVARPAIR;

/*
 * Local methods
 */

/** returns TRUE iff both keys are equal */
static
SCIP_DECL_HASHKEYEQ(varPairsEqual)
{  /*lint --e{715}*/
   SCIP* scip;
   ROWVARPAIR* varpair1;
   ROWVARPAIR* varpair2;
   SCIP_Real ratio1;
   SCIP_Real ratio2;

   scip = (SCIP*) userptr;
   varpair1 = (ROWVARPAIR*) key1;
   varpair2 = (ROWVARPAIR*) key2;

   if( varpair1->varindex1 != varpair2->varindex1 )
      return FALSE;

   if( varpair1->varindex2 != varpair2->varindex2 )
      return FALSE;

   ratio1 = varpair1->varcoef2 / varpair1->varcoef1;
   ratio2 = varpair2->varcoef2 / varpair2->varcoef1;

   return SCIPisEQ(scip, ratio1, ratio2);
}

/** returns the hash value of the key */
static
SCIP_DECL_HASHKEYVAL(varPairHashval)
{  /*lint --e{715}*/
   ROWVARPAIR* varpair;

   varpair = (ROWVARPAIR*) key;

   return SCIPhashTwo(SCIPcombineTwoInt(varpair->varindex1, varpair->varindex2),
                      SCIPrealHashCode(varpair->varcoef2 / varpair->varcoef1));
}

/** try non-zero cancellation for given row */
static
SCIP_RETCODE cancelRow(
   SCIP*                 scip,               /**< SCIP datastructure */
   SCIP_MATRIX*          matrix,             /**< the constraint matrix */
   SCIP_HASHTABLE*       pairtable,          /**< the hashtable containing ROWVARPAIR's of equations */
   int                   rowidx,             /**< index of row to try non-zero cancellation for */
   int                   maxcontfillin,      /**< maximal fill-in allowed for continuous variables */
   int                   maxintfillin,       /**< maximal fill-in allowed for integral variables */
   int                   maxbinfillin,       /**< maximal fill-in allowed for binary variables */
   int                   maxconsiderednonzeros, /**< maximal number of non-zeros to consider for cancellation */
   SCIP_Bool             preserveintcoefs,   /**< only perform non-zero cancellation if integrality of coefficients is preserved? */
   SCIP_Longint*         nuseless,           /**< pointer to update number of useless hashtable retrieves */
   int*                  nchgcoefs,          /**< pointer to update number of changed coefficients */
   int*                  ncanceled,          /**< pointer to update number of canceled nonzeros */
   int*                  nfillin             /**< pointer to update the produced fill-in */
   )
{
   int* cancelrowinds;
   SCIP_Real* cancelrowvals;
   SCIP_Real cancellhs;
   SCIP_Real cancelrhs;
   SCIP_Real bestcancelrate;
   int* tmpinds;
   int* locks;
   SCIP_Real* tmpvals;
   int cancelrowlen;
   int* rowidxptr;
   SCIP_Real* rowvalptr;
   int nchgcoef;
   int nretrieves;
   int bestnfillin;
   SCIP_Real mincancelrate;
   SCIP_Bool rowiseq;
   SCIP_CONS* cancelcons;

   rowiseq = SCIPisEQ(scip, SCIPmatrixGetRowLhs(matrix, rowidx), SCIPmatrixGetRowRhs(matrix, rowidx));

   cancelrowlen = SCIPmatrixGetRowNNonzs(matrix, rowidx);
   rowidxptr = SCIPmatrixGetRowIdxPtr(matrix, rowidx);
   rowvalptr = SCIPmatrixGetRowValPtr(matrix, rowidx);

   cancelcons = SCIPmatrixGetCons(matrix, rowidx);

   mincancelrate = 0.0;

   /* for set packing and logicor constraints, only accept equalities where all modified coefficients are cancelled */
   if( SCIPconsGetHdlr(cancelcons) == SCIPfindConshdlr(scip, "setppc") ||
       SCIPconsGetHdlr(cancelcons) == SCIPfindConshdlr(scip, "logicor") )
      mincancelrate = 1.0;

   SCIP_CALL( SCIPduplicateBufferArray(scip, &cancelrowinds, rowidxptr, cancelrowlen) );
   SCIP_CALL( SCIPduplicateBufferArray(scip, &cancelrowvals, rowvalptr, cancelrowlen) );
   SCIP_CALL( SCIPallocBufferArray(scip, &tmpinds, cancelrowlen) );
   SCIP_CALL( SCIPallocBufferArray(scip, &tmpvals, cancelrowlen) );
   SCIP_CALL( SCIPallocBufferArray(scip, &locks, cancelrowlen) );

   cancellhs = SCIPmatrixGetRowLhs(matrix, rowidx);
   cancelrhs = SCIPmatrixGetRowRhs(matrix, rowidx);

   nchgcoef = 0;
   nretrieves = 0;
   while( TRUE ) /*lint !e716 */
   {
      SCIP_Real bestscale;
      int bestcand;
      int i;
      int j;
      ROWVARPAIR rowvarpair;
      int maxlen;

      bestscale = 1.0;
      bestcand = -1;
      bestnfillin = 0;
      bestcancelrate = 0.0;

      for( i = 0; i < cancelrowlen; ++i )
      {
         tmpinds[i] = i;
         locks[i] = SCIPmatrixGetColNDownlocks(matrix, cancelrowinds[i]) + SCIPmatrixGetColNUplocks(matrix, cancelrowinds[i]);
      }

      SCIPsortIntInt(locks, tmpinds, cancelrowlen);

      maxlen = cancelrowlen;
      if( maxconsiderednonzeros >= 0 )
         maxlen = MIN(cancelrowlen, maxconsiderednonzeros);

      for( i = 0; i < maxlen; ++i )
      {
         for( j = i + 1; j < maxlen; ++j )
         {
            int a,b;
            int ncancel;
            int ncontfillin;
            int nintfillin;
            int nbinfillin;
            int ntotfillin;
            int eqrowlen;
            ROWVARPAIR* eqrowvarpair;
            SCIP_Real* eqrowvals;
            int* eqrowinds;
            SCIP_Real scale;
            SCIP_Real cancelrate;
            int i1,i2;
            SCIP_Bool abortpair;

            i1 = tmpinds[i];
            i2 = tmpinds[j];

            assert(cancelrowinds[i] < cancelrowinds[j]);

            if( cancelrowinds[i1] < cancelrowinds[i2] )
            {
               rowvarpair.varindex1 = cancelrowinds[i1];
               rowvarpair.varindex2 = cancelrowinds[i2];
               rowvarpair.varcoef1 = cancelrowvals[i1];
               rowvarpair.varcoef2 = cancelrowvals[i2];
            }
            else
            {
               rowvarpair.varindex1 = cancelrowinds[i2];
               rowvarpair.varindex2 = cancelrowinds[i1];
               rowvarpair.varcoef1 = cancelrowvals[i2];
               rowvarpair.varcoef2 = cancelrowvals[i1];
            }

            eqrowvarpair = (ROWVARPAIR*)SCIPhashtableRetrieve(pairtable, (void*) &rowvarpair);
            nretrieves++;

            if( eqrowvarpair == NULL || eqrowvarpair->rowindex == rowidx )
               continue;

            /* if the row we want to cancel is an equality, we will only use equalities
             * for canceling with less non-zeros and if the number of non-zeros is equal we use the
             * rowindex as tie-breaker to avoid cyclic non-zero cancellation
             */
            eqrowlen = SCIPmatrixGetRowNNonzs(matrix, eqrowvarpair->rowindex);
            if( rowiseq && (cancelrowlen < eqrowlen || (cancelrowlen == eqrowlen && rowidx < eqrowvarpair->rowindex)) )
               continue;

            eqrowvals = SCIPmatrixGetRowValPtr(matrix, eqrowvarpair->rowindex);
            eqrowinds = SCIPmatrixGetRowIdxPtr(matrix, eqrowvarpair->rowindex);

            scale = -rowvarpair.varcoef1 / eqrowvarpair->varcoef1;

            if( REALABS(scale) > MAXSCALE )
               continue;

            a = 0;
            b = 0;
            ncancel = 0;

            ncontfillin = 0;
            nintfillin = 0;
            nbinfillin = 0;
            abortpair = FALSE;
            while( a < cancelrowlen && b < eqrowlen )
            {
               if( cancelrowinds[a] == eqrowinds[b] )
               {
                  SCIP_Real newcoef;

                  newcoef = cancelrowvals[a] + scale * eqrowvals[b];

                  /* check if coefficient is cancelled */
                  if( SCIPisZero(scip, newcoef) )
                  {
                     ++ncancel;
                  }
                  /* otherwise, check if integral coefficients are preserved if the column is integral */
                  else if( (preserveintcoefs && SCIPvarIsIntegral(SCIPmatrixGetVar(matrix, cancelrowinds[a])) &&
                            SCIPisIntegral(scip, cancelrowvals[a]) && !SCIPisIntegral(scip, newcoef)) )
                  {
                     abortpair = TRUE;
                     break;
                  }
                  /* finally, check if locks could be modified in a bad way due to flipped signs */
                  else if( (SCIPisInfinity(scip, cancelrhs) || SCIPisInfinity(scip, -cancellhs)) &&
                           COPYSIGN(1.0, newcoef) != COPYSIGN(1.0, cancelrowvals[a]) ) /*lint !e777*/
                  {
                     /* do not flip signs for non-canceled coefficients if this adds a lock to a variable that had at most one lock
                      * in that direction before, except if the other direction gets unlocked
                      */
                     if( (cancelrowvals[a] > 0.0 && ! SCIPisInfinity(scip, cancelrhs)) ||
                         (cancelrowvals[a] < 0.0 && ! SCIPisInfinity(scip, -cancellhs)) )
                     {
                        /* if we get into this case the variable had a positive coefficient in a <= constraint or a negative
                         * coefficient in a >= constraint, e.g. an uplock. If this was the only uplock we do not abort their
                         * cancelling, otherwise we abort if we had a single or no downlock and add one
                         */
                        if( SCIPmatrixGetColNUplocks(matrix, cancelrowinds[a]) > 1 &&
                            SCIPmatrixGetColNDownlocks(matrix, cancelrowinds[a]) <= 1 )
                        {
                           abortpair = TRUE;
                           break;
                        }
                     }

                     if( (cancelrowvals[a] < 0.0 && ! SCIPisInfinity(scip, cancelrhs)) ||
                         (cancelrowvals[a] > 0.0 && ! SCIPisInfinity(scip, -cancellhs)) )
                     {
                        /* symmetric case where the variable had a downlock */
                        if( SCIPmatrixGetColNDownlocks(matrix, cancelrowinds[a]) > 1 &&
                            SCIPmatrixGetColNUplocks(matrix, cancelrowinds[a]) <= 1 )
                        {
                           abortpair = TRUE;
                           break;
                        }
                     }
                  }

                  ++a;
                  ++b;
               }
               else if( cancelrowinds[a] < eqrowinds[b] )
               {
                  ++a;
               }
               else
               {
                  SCIP_Real newcoef;
                  SCIP_VAR* var;

                  var = SCIPmatrixGetVar(matrix, eqrowinds[b]);
                  newcoef = scale * eqrowvals[b];

                  if( (newcoef > 0.0 && ! SCIPisInfinity(scip, cancelrhs)) ||
                      (newcoef < 0.0 && ! SCIPisInfinity(scip, -cancellhs)) )
                  {
                     if( SCIPmatrixGetColNUplocks(matrix, eqrowinds[b]) <= 1 )
                     {
                        abortpair = TRUE;
                        ++b;
                        break;
                     }
                  }

                  if( (newcoef < 0.0 && ! SCIPisInfinity(scip, cancelrhs)) ||
                      (newcoef > 0.0 && ! SCIPisInfinity(scip, -cancellhs)) )
                  {
                     if( SCIPmatrixGetColNDownlocks(matrix, eqrowinds[b]) <= 1 )
                     {
                        abortpair = TRUE;
                        ++b;
                        break;
                     }
                  }

                  ++b;

                  if( SCIPvarIsIntegral(var) )
                  {
                     if( SCIPvarIsBinary(var) && ++nbinfillin > maxbinfillin )
                     {
                        abortpair = TRUE;
                        break;
                     }

                     if( ++nintfillin > maxintfillin )
                     {
                        abortpair = TRUE;
                        break;
                     }
                  }
                  else
                  {
                     if( ++ncontfillin > maxcontfillin )
                     {
                        abortpair = TRUE;
                        break;
                     }
                  }
               }
            }

            if( abortpair )
               continue;

            while( b < eqrowlen )
            {
               SCIP_VAR* var = SCIPmatrixGetVar(matrix, eqrowinds[b]);
               ++b;
               if( SCIPvarIsIntegral(var) )
               {
                  if( SCIPvarIsBinary(var) && ++nbinfillin > maxbinfillin )
                     break;
                  if( ++nintfillin > maxintfillin )
                     break;
               }
               else
               {
                  if( ++ncontfillin > maxcontfillin )
                     break;
               }
            }

            if( ncontfillin > maxcontfillin || nbinfillin > maxbinfillin || nintfillin > maxintfillin )
               continue;

            ntotfillin = nintfillin + ncontfillin;

            if( ntotfillin >= ncancel )
               continue;

            cancelrate = (ncancel - ntotfillin) / (SCIP_Real) eqrowlen;

            if( cancelrate < mincancelrate )
               continue;

            if( cancelrate > bestcancelrate )
            {
               bestnfillin = ntotfillin;
               bestcand = eqrowvarpair->rowindex;
               bestscale = scale;
               bestcancelrate = cancelrate;

               /* stop looking if the current candidate does not create any fill-in or alter coefficients */
               if( cancelrate == 1.0 )
                  break;
            }

            /* we accept the best candidate immediately if it does not create any fill-in or alter coefficients */
            if( bestcand != -1 && bestcancelrate == 1.0 )
               break;
         }
      }

      if( bestcand != -1 )
      {
         int a;
         int b;
         SCIP_Real* eqrowvals;
         int* eqrowinds;
         int eqrowlen;
         int tmprowlen;
         SCIP_Real eqrhs;

         eqrowvals = SCIPmatrixGetRowValPtr(matrix, bestcand);
         eqrowinds = SCIPmatrixGetRowIdxPtr(matrix, bestcand);
         eqrowlen = SCIPmatrixGetRowNNonzs(matrix, bestcand);
         eqrhs = SCIPmatrixGetRowRhs(matrix, bestcand);

         a = 0;
         b = 0;
         tmprowlen = 0;

         if( !SCIPisZero(scip, eqrhs) )
         {
            if( !SCIPisInfinity(scip, -cancellhs) )
               cancellhs += bestscale * eqrhs;
            if( !SCIPisInfinity(scip, cancelrhs) )
               cancelrhs += bestscale * eqrhs;
         }

         while( a < cancelrowlen && b < eqrowlen )
         {
            if( cancelrowinds[a] == eqrowinds[b] )
            {
               SCIP_Real val = cancelrowvals[a] + bestscale * eqrowvals[b];

               if( !SCIPisZero(scip, val) )
               {
                  tmpinds[tmprowlen] = cancelrowinds[a];
                  tmpvals[tmprowlen] = val;
                  ++tmprowlen;
               }
               ++nchgcoef;

               ++a;
               ++b;
            }
            else if( cancelrowinds[a] < eqrowinds[b] )
            {
               tmpinds[tmprowlen] = cancelrowinds[a];
               tmpvals[tmprowlen] = cancelrowvals[a];
               ++tmprowlen;
               ++a;
            }
            else
            {
               tmpinds[tmprowlen] = eqrowinds[b];
               tmpvals[tmprowlen] = eqrowvals[b] * bestscale;
               ++nchgcoef;
               ++tmprowlen;
               ++b;
            }
         }

         while( a < cancelrowlen )
         {
            tmpinds[tmprowlen] = cancelrowinds[a];
            tmpvals[tmprowlen] = cancelrowvals[a];
            ++tmprowlen;
            ++a;
         }

         while( b < eqrowlen )
         {
            tmpinds[tmprowlen] = eqrowinds[b];
            tmpvals[tmprowlen] = eqrowvals[b] * bestscale;
            ++nchgcoef;
            ++tmprowlen;
            ++b;
         }

         /* update fill-in counter */
         *nfillin += bestnfillin;

         /* swap the temporary arrays so that the cancelrowinds and cancelrowvals arrays, contain the new
          * changed row, and the tmpinds and tmpvals arrays can be overwritten in the next iteration
          */
         SCIPswapPointers((void**) &tmpinds, (void**) &cancelrowinds);
         SCIPswapPointers((void**) &tmpvals, (void**) &cancelrowvals);
         cancelrowlen = tmprowlen;
      }
      else
         break;
   }

   if( nchgcoef != 0 )
   {
      SCIP_CONS* cons;
      SCIP_VAR** consvars;

      int i;

      SCIP_CALL( SCIPallocBufferArray(scip, &consvars, cancelrowlen) );

      for( i = 0; i < cancelrowlen; ++i )
         consvars[i] = SCIPmatrixGetVar(matrix, cancelrowinds[i]);

      /* create sparsified constraint and add it to scip */
      SCIP_CALL( SCIPcreateConsLinear(scip, &cons, SCIPmatrixGetRowName(matrix, rowidx), cancelrowlen, consvars, cancelrowvals,
                                      cancellhs, cancelrhs, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
      SCIP_CALL( SCIPdelCons(scip, SCIPmatrixGetCons(matrix, rowidx)) );
      SCIP_CALL( SCIPaddCons(scip, cons) );

#ifdef SCIP_MORE_DEBUG
      SCIPdebugMsg(scip, "########\n");
      SCIPdebugMsg(scip, "old:\n");
      SCIPmatrixPrintRow(scip, matrix, rowidx);
      SCIPdebugMsg(scip, "new:\n");
      SCIPdebugPrintCons(scip, cons, NULL);
      SCIPdebugMsg(scip, "########\n");
#endif

      SCIP_CALL( SCIPreleaseCons(scip, &cons) );

      /* update counters */
      *nchgcoefs += nchgcoef;
      *ncanceled += SCIPmatrixGetRowNNonzs(matrix, rowidx) - cancelrowlen;

      /* if successful, decrease the useless hashtable retrieves counter; the rationale here is that we want to keep
       * going if, after many useless calls that almost exceeded the budget, we finally reach a useful section; but we
       * don't allow a negative build-up for the case that the useful section is all at the beginning and we just want
       * to quit quickly afterwards
       */
      *nuseless -= nretrieves;
      *nuseless = MAX(*nuseless, 0);

      SCIPfreeBufferArray(scip, &consvars);
   }
   else
   {
      /* if not successful, increase useless hashtable retrieves counter */
      *nuseless += nretrieves;
   }

   SCIPfreeBufferArray(scip, &locks);
   SCIPfreeBufferArray(scip, &tmpvals);
   SCIPfreeBufferArray(scip, &tmpinds);
   SCIPfreeBufferArray(scip, &cancelrowvals);
   SCIPfreeBufferArray(scip, &cancelrowinds);

   return SCIP_OKAY;
}


/** try to concel the nonzeros in two variables */
static
void cancelCol(
   SCIP*                 scip,               /**< SCIP datastructure */
   SCIP_MATRIX*          matrix,             /**< the constraint matrix */
   int                   rowidxi,            /**< one of the indexes of column to try non-zero cancellation for */
   int                   rowidxj,            /**< one of the indexes of column to try non-zero cancellation for */
   int                   success,            /**< does cancellation success? 0: failed, 1: cancel variable j, 2: cancel both variables i and j */
   int*                  ratios,             /**< ratio of the vectors*/
   int*                  nratios,            /**< number of different ratios*/
   int*                  nchgcoefs,          /**< pointer to update number of changed coefficients */
   int*                  ncanceled,          /**< pointer to update number of canceled nonzeros */
   int*                  nfillin             /**< pointer to update the produced fill-in */
      )
{

}

/** updates failure counter after one execution */
static
void updateFailureStatistic(
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   SCIP_Bool             success             /**< was this execution successful? */
   )
{
   assert(presoldata != NULL);

   if( success )
   {
      presoldata->nfailures = 0;
      presoldata->nwaitingcalls = 0;
   }
   else
   {
      presoldata->nfailures++;
      presoldata->nwaitingcalls = (int)(presoldata->waitingfac*(SCIP_Real)presoldata->nfailures);
   }
}


/*
 * Callback methods of presolver
 */

/** copy method for constraint handler plugins (called when SCIP copies plugins) */
static
SCIP_DECL_PRESOLCOPY(presolCopyDualsparsify)
{
   SCIP_PRESOLDATA* presoldata;

   assert(scip != NULL);
   assert(presol != NULL);
   assert(strcmp(SCIPpresolGetName(presol), PRESOL_NAME) == 0);

   /* call inclusion method of presolver if copying is enabled */
   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);
   if( presoldata->enablecopy )
   {
      SCIP_CALL( SCIPincludePresolDualsparsify(scip) );
   }

   return SCIP_OKAY;
}

/** execution method of presolver */
static
SCIP_DECL_PRESOLEXEC(presolExecDualsparsify)
{  /*lint --e{715}*/
   SCIP_MATRIX* matrix;
   SCIP_Bool initialized;
   SCIP_Bool complete;
   int ncols;
   int r;
   int i;
   int j;
   int numcancel;
   int oldnchgcoefs;
   int nfillin;
   int* locks;
   int* perm;
   int* rowidxsorted;
   int* rowsparsity;
   int nvarpairs;
   int varpairssize;
   SCIP_PRESOLDATA* presoldata;
   SCIP_Longint maxuseless;
   SCIP_Longint nuseless;

   int *processedvarsidx;
   int nprocessedvarsidx;
   nprocessedvarsidx = 0;

   assert(result != NULL);

   *result = SCIP_DIDNOTRUN;

   if( (SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING) || SCIPinProbing(scip) || SCIPisNLPEnabled(scip) )
      return SCIP_OKAY;

   if( SCIPisStopped(scip) || SCIPgetNActivePricers(scip) > 0 )
      return SCIP_OKAY;

   presoldata = SCIPpresolGetData(presol);

   if( presoldata->nwaitingcalls > 0 )
   {
      presoldata->nwaitingcalls--;
      SCIPdebugMsg(scip, "skipping dualsparsify: nfailures=%d, nwaitingcalls=%d\n", presoldata->nfailures,
         presoldata->nwaitingcalls);
      return SCIP_OKAY;
   }
   SCIPdebugMsg(scip, "starting dualsparsify. . .\n");
   *result = SCIP_DIDNOTFIND;

   matrix = NULL;
   SCIP_CALL( SCIPmatrixCreate(scip, &matrix, &initialized, &complete) );
   if( initialized && complete )
   {
      SCIP_Real *ratios;
      int nratios;
      nratios = 0;
      ncols = SCIPmatrixGetNColumns(matrix);
      SCIP_CALL( SCIPallocBufferArray(scip, &processedvarsidx, ncols) );
      SCIP_CALL( SCIPallocBufferArray(scip, &ratios, ncols) );
      for(i=0; i<ncols; i++)
      {
         int nnonz;
         nnonz = SCIPmatrixGetColNNonzs(matrix, i);
         if(nnonz > 10 && SCIPvarGetType(SCIPmatrixGetVar(matrix, i)) == SCIP_VARTYPE_CONTINUOUS
               && !SCIPdoNotMultaggrVar(scip, SCIPmatrixGetVar(matrix, i)))
         {
            processedvarsidx[nprocessedvarsidx] = i;
            nprocessedvarsidx++;
         }
      }

      for(i=0; i<nprocessedvarsidx; i++)
      {
         int* colpnt = SCIPmatrixGetColIdxPtr(matrix, processedvarsidx[i]);
         SCIP_Real* valpnt = SCIPmatrixGetColValPtr(matrix, processedvarsidx[i]);
         SCIPsortIntReal(colpnt, valpnt, SCIPmatrixGetColNNonzs(matrix, processedvarsidx[i]));
         printf("%s, %d\n", SCIPmatrixGetColName(matrix, processedvarsidx[i]),
               SCIPmatrixGetColNNonzs(matrix, processedvarsidx[i]));
      }

      for(i=0; i<nprocessedvarsidx; i++)
      {
         int* colpnti = SCIPmatrixGetColIdxPtr(matrix, processedvarsidx[i]);
         SCIP_Real* valpnti = SCIPmatrixGetColValPtr(matrix, processedvarsidx[i]);
         SCIP_VAR* vari;
         SCIP_VAR* varj;
         SCIP_VAR* newvar;
         SCIP_CONS* newcons;
         SCIP_VAR* vars[2];
         SCIP_Real coefs[2];


         vari = SCIPmatrixGetVar(matrix, processedvarsidx[i]);
         SCIP_Real newlb;
         SCIP_Real newub;

         for(j=i+1; j<nprocessedvarsidx; j++)
         {
            SCIP_Bool infeasible;
            SCIP_Bool aggregated;
            varj = SCIPmatrixGetVar(matrix, processedvarsidx[j]);
            int* colpntj = SCIPmatrixGetColIdxPtr(matrix, processedvarsidx[j]);
            SCIP_Real* valpntj = SCIPmatrixGetColValPtr(matrix, processedvarsidx[j]);
            char newvarname[SCIP_MAXSTRLEN];
            (void) SCIPsnprintf(newvarname, SCIP_MAXSTRLEN, "%s_agg_%s",
                  SCIPvarGetName(vari), SCIPvarGetName(varj));
            newlb = SCIPmatrixGetColLb(matrix, processedvarsidx[i]) + SCIPmatrixGetColLb(matrix, processedvarsidx[j]);
            newub = SCIPmatrixGetColUb(matrix, processedvarsidx[i]) + SCIPmatrixGetColUb(matrix, processedvarsidx[j]);   
            SCIP_CALL( SCIPcreateVar(scip, &newvar, newvarname, newlb, newub, 0.0, SCIP_VARTYPE_CONTINUOUS,
                     SCIPvarIsInitial(varj), SCIPvarIsRemovable(varj), NULL, NULL, NULL, NULL, NULL) );
            SCIP_CALL( SCIPaddVar(scip, newvar) );
            vars[0] = newvar;
            vars[1] = vari;
            coefs[0] = 1;
            coefs[1] = -1;

            SCIP_CALL( SCIPmultiaggregateVar(scip, varj, 2, vars, coefs, 0.0, &infeasible, &aggregated) );
            assert(!infeasible);
            assert(aggregated);
//            printf("%8.4f\n", newvar)

            char newconsname[SCIP_MAXSTRLEN];
            (void) SCIPsnprintf(newconsname, SCIP_MAXSTRLEN, "%s_dual_%s",
                  SCIPvarGetName(vari), SCIPvarGetName(varj));
            SCIP_CALL( SCIPcreateConsVarbound(scip, &newcons, newconsname, vars[0], vars[1], coefs[1],
                     SCIPmatrixGetColLb(matrix, processedvarsidx[j]), SCIPmatrixGetColUb(matrix, processedvarsidx[j]),
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
            SCIP_CALL( SCIPaddCons(scip, newcons) );
            SCIPdebugPrintCons(scip, newcons, NULL);
            SCIP_CALL( SCIPreleaseCons(scip, &newcons) );
            SCIP_CALL( SCIPreleaseVar(scip, &newvar) );
         }
      }

      SCIPfreeBufferArray(scip, processedvarsidx);
   }

#if 0
   /* if we want to cancel only from specialized constraints according to the parameter, then we can skip execution if
    * only linear constraints are present
    */
   linearhdlr = SCIPfindConshdlr(scip, "linear");
   if( !presoldata->cancellinear && linearhdlr != NULL && SCIPconshdlrGetNConss(linearhdlr) >= SCIPgetNConss(scip) )
   {
      SCIPdebugMsg(scip, "skipping dualsparsify: only linear constraints found\n");
      return SCIP_OKAY;
   }

   SCIPdebugMsg(scip, "starting dualsparsify. . .\n");
   *result = SCIP_DIDNOTFIND;

   matrix = NULL;
   SCIP_CALL( SCIPmatrixCreate(scip, &matrix, &initialized, &complete) );

   /* we only work on pure MIPs currently */
   if( initialized && complete )
   {
      nrows = SCIPmatrixGetNRows(matrix);

      /* sort rows by column indices */
      for( i = 0; i < nrows; i++ )
      {
         int* rowpnt = SCIPmatrixGetRowIdxPtr(matrix, i);
         SCIP_Real* valpnt = SCIPmatrixGetRowValPtr(matrix, i);
         SCIPsortIntReal(rowpnt, valpnt, SCIPmatrixGetRowNNonzs(matrix, i));
      }

      SCIP_CALL( SCIPallocBufferArray(scip, &locks, SCIPmatrixGetNColumns(matrix)) );
      SCIP_CALL( SCIPallocBufferArray(scip, &perm, SCIPmatrixGetNColumns(matrix)) );

      /* loop over all rows and create var pairs */
      numcancel = 0;
      nfillin = 0;
      varpairssize = 0;
      nvarpairs = 0;
      varpairs = NULL;
      SCIP_CALL( SCIPhashtableCreate(&pairtable, SCIPblkmem(scip), 1, SCIPhashGetKeyStandard, varPairsEqual, varPairHashval, (void*) scip) );

      /* collect equalities and their number of non-zeros */
      for( r = 0; r < nrows; r++ )
      {
         int nnonz;

         nnonz = SCIPmatrixGetRowNNonzs(matrix, r);

         /* consider equalities with support at most maxnonzeros; skip singleton equalities, because these are faster
          * processed by trivial presolving
          */
         if( nnonz >= 2 && (presoldata->maxnonzeros < 0 || nnonz <= presoldata->maxnonzeros)
            && SCIPisEQ(scip, SCIPmatrixGetRowRhs(matrix, r), SCIPmatrixGetRowLhs(matrix, r)) )
         {
            int* rowinds;
            SCIP_Real* rowvals;
            int npairs;
            int failshift;

            rowinds = SCIPmatrixGetRowIdxPtr(matrix, r);
            rowvals = SCIPmatrixGetRowValPtr(matrix, r);

            for( i = 0; i < nnonz; ++i )
            {
               perm[i] = i;
               locks[i] = SCIPmatrixGetColNDownlocks(matrix, rowinds[i]) + SCIPmatrixGetColNUplocks(matrix, rowinds[i]);
            }

            SCIPsortIntInt(locks, perm, nnonz);

            if( presoldata->maxconsiderednonzeros >= 0 )
               nnonz = MIN(nnonz, presoldata->maxconsiderednonzeros);

            npairs = (nnonz * (nnonz - 1)) / 2;
            if( nvarpairs + npairs > varpairssize )
            {
               int newsize = SCIPcalcMemGrowSize(scip, nvarpairs + npairs);
               SCIP_CALL( SCIPreallocBufferArray(scip, &varpairs, newsize) );
               varpairssize = newsize;
            }

            /* if we are called after one or more failures, i.e., executions without finding cancellations, then we
             * shift the section of nonzeros considered; in the case that the maxconsiderednonzeros limit is hit, this
             * results in different variable pairs being tried and avoids trying the same useless cancellations
             * repeatedly
             */
            failshift = presoldata->nfailures*presoldata->maxconsiderednonzeros;

            for( i = 0; i < nnonz; ++i )
            {
               for( j = i + 1; j < nnonz; ++j )
               {
                  int i1;
                  int i2;

                  assert(nvarpairs < varpairssize);
                  assert(varpairs != NULL);

                  i1 = perm[(i + failshift) % nnonz];
                  i2 = perm[(j + failshift) % nnonz];
                  varpairs[nvarpairs].rowindex = r;

                  if( rowinds[i1] < rowinds[i2])
                  {
                     varpairs[nvarpairs].varindex1 = rowinds[i1];
                     varpairs[nvarpairs].varindex2 = rowinds[i2];
                     varpairs[nvarpairs].varcoef1 = rowvals[i1];
                     varpairs[nvarpairs].varcoef2 = rowvals[i2];
                  }
                  else
                  {
                     varpairs[nvarpairs].varindex1 = rowinds[i2];
                     varpairs[nvarpairs].varindex2 = rowinds[i1];
                     varpairs[nvarpairs].varcoef1 = rowvals[i2];
                     varpairs[nvarpairs].varcoef2 = rowvals[i1];
                  }
                  ++nvarpairs;
               }
            }
         }
      }

      /* insert varpairs into hash table */
      for( r = 0; r < nvarpairs; ++r )
      {
         SCIP_Bool insert;
         ROWVARPAIR* othervarpair;

         assert(varpairs != NULL);

         insert = TRUE;

         /* check if this pair is already contained in the hash table;
          * The loop is required due to the non-transitivity of the hash functions
          */
         while( (othervarpair = (ROWVARPAIR*)SCIPhashtableRetrieve(pairtable, (void*) &varpairs[r])) != NULL )
         {
            /* if the previous variable pair has fewer or the same number of non-zeros in the attached row
             * we keep that pair and skip this one
             */
            if( SCIPmatrixGetRowNNonzs(matrix, othervarpair->rowindex) <= SCIPmatrixGetRowNNonzs(matrix, varpairs[r].rowindex) )
            {
               insert = FALSE;
               break;
            }

            /* this pairs row has fewer non-zeros, so remove the other pair from the hash table and loop */
            SCIP_CALL( SCIPhashtableRemove(pairtable, (void*) othervarpair) );
         }

         if( insert )
         {
            SCIP_CALL( SCIPhashtableInsert(pairtable, (void*) &varpairs[r]) );
         }
      }

      /* sort rows according to parameter value */
      if( presoldata->rowsort == 'i' || presoldata->rowsort == 'd' )
      {
         SCIP_CALL( SCIPallocBufferArray(scip, &rowidxsorted, nrows) );
         SCIP_CALL( SCIPallocBufferArray(scip, &rowsparsity, nrows) );
         for( r = 0; r < nrows; ++r )
            rowidxsorted[r] = r;
         if( presoldata->rowsort == 'i' )
         {
            for( r = 0; r < nrows; ++r )
               rowsparsity[r] = SCIPmatrixGetRowNNonzs(matrix, r);
         }
         else if( presoldata->rowsort == 'd' )
         {
            for( r = 0; r < nrows; ++r )
               rowsparsity[r] = -SCIPmatrixGetRowNNonzs(matrix, r);
         }
         SCIPsortIntInt(rowsparsity, rowidxsorted, nrows);
      }
      else
      {
         assert(presoldata->rowsort == 'n');
         rowidxsorted = NULL;
         rowsparsity = NULL;
      }

      /* loop over the rows and cancel non-zeros until maximum number of retrieves is reached */
      maxuseless = (SCIP_Longint)(presoldata->maxretrievefac * (SCIP_Real)nrows);
      nuseless = 0;
      oldnchgcoefs = *nchgcoefs;
      for( r = 0; r < nrows && nuseless <= maxuseless; r++ )
      {
         int rowidx;

         rowidx = rowidxsorted != NULL ? rowidxsorted[r] : r;

         /* check whether we want to cancel only from specialized constraints; one reasoning behind this may be that
          * cancelling fractional coefficients requires more numerical care than is currently implemented in method
          * cancelRow()
          */
         assert(SCIPmatrixGetCons(matrix, rowidx) != NULL);
         if( !presoldata->cancellinear && SCIPconsGetHdlr(SCIPmatrixGetCons(matrix, rowidx)) == linearhdlr )
            continue;

         /* since the function parameters for the max fillin are unsigned we do not need to handle the
          * unlimited (-1) case due to implicit conversion rules */
         SCIP_CALL( cancelRow(scip, matrix, pairtable, rowidx, \
               presoldata->maxcontfillin == -1 ? INT_MAX : presoldata->maxcontfillin, \
               presoldata->maxintfillin == -1 ? INT_MAX : presoldata->maxintfillin, \
               presoldata->maxbinfillin == -1 ? INT_MAX : presoldata->maxbinfillin, \
               presoldata->maxconsiderednonzeros, presoldata->preserveintcoefs, \
               &nuseless, nchgcoefs, &numcancel, &nfillin) );
      }

      SCIPfreeBufferArrayNull(scip, &rowsparsity);
      SCIPfreeBufferArrayNull(scip, &rowidxsorted);

      SCIPhashtableFree(&pairtable);
      SCIPfreeBufferArrayNull(scip, &varpairs);

      SCIPfreeBufferArray(scip, &perm);
      SCIPfreeBufferArray(scip, &locks);

      /* update result */
      presoldata->ncancels += numcancel;
      presoldata->nfillin += nfillin;

      if( numcancel > 0 )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL,
            "   (%.1fs) dualsparsify %s: %d/%d (%.1f%%) nonzeros canceled"
            " - in total %d canceled nonzeros, %d changed coefficients, %d added nonzeros\n",
            SCIPgetSolvingTime(scip), (nuseless > maxuseless ? "aborted" : "finished"), numcancel,
            SCIPmatrixGetNNonzs(matrix), 100.0*(SCIP_Real)numcancel/(SCIP_Real)SCIPmatrixGetNNonzs(matrix),
            presoldata->ncancels, SCIPpresolGetNChgCoefs(presol) + *nchgcoefs - oldnchgcoefs, presoldata->nfillin);
         *result = SCIP_SUCCESS;
      }

      updateFailureStatistic(presoldata, numcancel > 0);

      SCIPdebugMsg(scip, "dualsparsify failure statistic: nfailures=%d, nwaitingcalls=%d\n", presoldata->nfailures,
         presoldata->nwaitingcalls);
   }
   /* if matrix construction fails once, we do not ever want to be called again */
   else
   {
      updateFailureStatistic(presoldata, FALSE);
      presoldata->nwaitingcalls = INT_MAX;
   }
#endif
   SCIPmatrixFree(scip, &matrix);

   return SCIP_OKAY;
}

/*
 * presolver specific interface methods
 */

/** destructor of presolver to free user data (called when SCIP is exiting) */
static
SCIP_DECL_PRESOLFREE(presolFreeDualsparsify)
{  /*lint --e{715}*/
   SCIP_PRESOLDATA* presoldata;

   /* free presolver data */
   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);

   SCIPfreeBlockMemory(scip, &presoldata);
   SCIPpresolSetData(presol, NULL);

   return SCIP_OKAY;
}

/** initialization method of presolver (called after problem was transformed) */
static
SCIP_DECL_PRESOLINIT(presolInitDualsparsify)
{
   SCIP_PRESOLDATA* presoldata;

   /* set the counters in the init (and not in the initpre) callback such that they persist across restarts */
   presoldata = SCIPpresolGetData(presol);
   presoldata->ncancels = 0;
   presoldata->nfillin = 0;
   presoldata->nfailures = 0;
   presoldata->nwaitingcalls = 0;

   return SCIP_OKAY;
}

/** creates the dualsparsify presolver and includes it in SCIP */
SCIP_RETCODE SCIPincludePresolDualsparsify(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_PRESOLDATA* presoldata;
   SCIP_PRESOL* presol;

   /* create dualsparsify presolver data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &presoldata) );

   /* include presolver */
   SCIP_CALL( SCIPincludePresolBasic(scip, &presol, PRESOL_NAME, PRESOL_DESC, PRESOL_PRIORITY, PRESOL_MAXROUNDS,
         PRESOL_TIMING, presolExecDualsparsify, presoldata) );

   SCIP_CALL( SCIPsetPresolCopy(scip, presol, presolCopyDualsparsify) );
   SCIP_CALL( SCIPsetPresolFree(scip, presol, presolFreeDualsparsify) );
   SCIP_CALL( SCIPsetPresolInit(scip, presol, presolInitDualsparsify) );

   SCIP_CALL( SCIPaddBoolParam(scip,
         "presolving/dualsparsify/enablecopy",
         "should dualsparsify presolver be copied to sub-SCIPs?",
         &presoldata->enablecopy, TRUE, DEFAULT_ENABLECOPY, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
         "presolving/dualsparsify/cancellinear",
         "should we cancel nonzeros in constraints of the linear constraint handler?",
         &presoldata->cancellinear, TRUE, DEFAULT_CANCELLINEAR, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
         "presolving/dualsparsify/preserveintcoefs",
         "should we forbid cancellations that destroy integer coefficients?",
         &presoldata->preserveintcoefs, TRUE, DEFAULT_PRESERVEINTCOEFS, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
         "presolving/dualsparsify/maxcontfillin",
         "maximal fillin for continuous variables (-1: unlimited)",
         &presoldata->maxcontfillin, FALSE, DEFAULT_MAX_CONT_FILLIN, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
         "presolving/dualsparsify/maxbinfillin",
         "maximal fillin for binary variables (-1: unlimited)",
         &presoldata->maxbinfillin, FALSE, DEFAULT_MAX_BIN_FILLIN, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
         "presolving/dualsparsify/maxintfillin",
         "maximal fillin for integer variables including binaries (-1: unlimited)",
         &presoldata->maxintfillin, FALSE, DEFAULT_MAX_INT_FILLIN, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
         "presolving/dualsparsify/maxnonzeros",
         "maximal support of one equality to be used for cancelling (-1: no limit)",
         &presoldata->maxnonzeros, TRUE, DEFAULT_MAXNONZEROS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
         "presolving/dualsparsify/maxconsiderednonzeros",
         "maximal number of considered non-zeros within one row (-1: no limit)",
         &presoldata->maxconsiderednonzeros, TRUE, DEFAULT_MAXCONSIDEREDNONZEROS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddCharParam(scip,
         "presolving/dualsparsify/rowsort",
         "order in which to process inequalities ('n'o sorting, 'i'ncreasing nonzeros, 'd'ecreasing nonzeros)",
         &presoldata->rowsort, TRUE, DEFAULT_ROWSORT, "nid", NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip,
         "presolving/dualsparsify/maxretrievefac",
         "limit on the number of useless vs. useful hashtable retrieves as a multiple of the number of constraints",
         &presoldata->maxretrievefac, TRUE, DEFAULT_MAXRETRIEVEFAC, 0.0, SCIP_REAL_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip,
         "presolving/dualsparsify/waitingfac",
         "number of calls to wait until next execution as a multiple of the number of useless calls",
         &presoldata->waitingfac, TRUE, DEFAULT_WAITINGFAC, 0.0, SCIP_REAL_MAX, NULL, NULL) );

   return SCIP_OKAY;
}
