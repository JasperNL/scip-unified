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

/**@file   sepa_rlt.c
 * @brief  RLT separator
 * @author Fabian Wegscheider
 *
 * @todo implement the possibility to add extra auxiliary variables for RLT (like in DOI 10.1080/10556788.2014.916287)
 * @todo add RLT cuts for the product of equality constraints
 * @todo implement dynamic addition of RLT cuts during branching (see DOI 10.1007/s10898-012-9874-7)
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/sepa_rlt.h"
#include "scip/cons_expr.h"
#include "cons_linear.h"
#include "cons_knapsack.h"
#include "cons_varbound.h"
#include "cons_setppc.h"
#include "cons_expr_iterator.h"
#include "cons_expr_pow.h"


#define SEPA_NAME              "rlt"
#define SEPA_DESC              "rlt separator"
#define SEPA_PRIORITY                10
#define SEPA_FREQ                     0
#define SEPA_MAXBOUNDDIST           1.0
#define SEPA_USESSUBSCIP          FALSE /**< does the separator use a secondary SCIP instance? */
#define SEPA_DELAY                FALSE /**< should separation method be delayed, if other separators found cuts? */

#define DEFAULT_MAXUNKNOWNTERMS       0 /**< default value for parameter maxunknownterms */
#define DEFAULT_MAXUSEDVARS         100 /**< default value for parameter maxusedvars */
#define DEFAULT_MAXNONZEROPROP      0.0 /**< default value for parameter maxnonzeroprop */
#define DEFAULT_MAXNCUTS             -1 /**< default value for parameter maxncuts */
#define DEFAULT_MAXROUNDS             1 /**< default value for parameter maxrounds */
#define DEFAULT_MAXROUNDSROOT        10 /**< default value for parameter maxroundsroot */
#define DEFAULT_ONLYEQROWS         TRUE /**< default value for parameter eqrowsfirst */
#define DEFAULT_ONLYCONTROWS       TRUE /**< default value for parameter eqrowsfirst */
#define DEFAULT_ONLYINITIAL        TRUE /**< default value for parameter onlyinitial */
#define DEFAULT_USEINSUBSCIP      FALSE /**< default value for parameter useinsubscip */

#define MAXVARBOUND                1e+5 /**< maximum allowed variable bound for computing an RLT-cut */

/*
 * Data structures
 */

/** separator data */
struct SCIP_SepaData
{
   SCIP_CONSHDLR*        conshdlr;           /**< expression constraint handler */
   SCIP_VAR**            varssorted;         /**< variables that occur in bilinear terms sorted by priority */
   SCIP_VAR**            bilinauxvars;       /**< linearization variable for each bilinear term */
   SCIP_HASHMAP*         bilinvarsmap;       /**< map for accessing the linearization variables of each bilinear term */
   int*                  varpriorities;      /**< priorities of the variables in varssorted */
   int                   maxvarindex;        /**< maximum variable index when creating bilinvarsmap */
   int                   nbilinterms;        /**< total number of bilinear terms */
   int                   nbilinvars;         /**< total number of variables occurring in bilinear terms */
   int                   currentnunknown;    /**< number of unknown terms in current row (not printed) */
   SCIP_Bool             iscreated;          /**< indicates whether the sepadata has been initialized yet */
   SCIP_Bool             isinitialround;     /**< indicates that this is the first round and initial rows are used */

   /* parameters */
   SCIP_Real             maxnonzeroprop;     /**< maximum acceptable proportion of known bilinear terms to non-zeroes */
   int                   maxunknownterms;    /**< maximum number of unknown bilinear terms a row can have to be used */
   int                   maxusedvars;        /**< maximum number of variables that will be used to compute rlt cuts */
   int                   maxncuts;           /**< maximum number of cuts that will be added per round */
   int                   maxrounds;          /**< maximal number of separation rounds per node (-1: unlimited) */
   int                   maxroundsroot;      /**< maximal number of separation rounds in the root node (-1: unlimited) */
   SCIP_Bool             onlyeqrows;         /**< indicates wether only equality rows should be used for rlt cuts */
   SCIP_Bool             onlycontrows;       /**< indicates wether only continuous rows should be used for rlt cuts */
   SCIP_Bool             onlyinitial;        /**< indicates whether only initial rows should be uswed for rlt cuts */
   SCIP_Bool             useinsubscip;       /**< indicates whether the seperator should also be used in sub-scips */
};

/*
 * Local methods
 */

/* helper method to free the separation data */
static
SCIP_RETCODE freeSepaData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata            /**< separation data */
   )
{  /*lint --e{715}*/
   int i;

   assert(sepadata->iscreated);
   assert(sepadata->bilinvarsmap != NULL);

   /* release auxiliary variables that were captured for rlt */
   for( i = 0; i < sepadata->nbilinterms; ++i )
   {
      assert(sepadata->bilinauxvars[i] != NULL);
      SCIP_CALL( SCIPreleaseVar(scip, &(sepadata->bilinauxvars[i])) );
   }

   /* release bilinvars that were captured for rlt */
   for( i = 0; i < sepadata->nbilinvars; ++i )
   {
      assert(sepadata->varssorted[i] != NULL);
      SCIP_CALL( SCIPreleaseVar(scip, &(sepadata->varssorted[i])) );
   }

   /* free arrays */
   SCIPfreeBlockMemoryArray(scip, &sepadata->bilinauxvars, sepadata->nbilinterms);
   SCIPfreeBlockMemoryArray(scip, &sepadata->varpriorities, sepadata->nbilinvars);
   SCIPfreeBlockMemoryArray(scip, &sepadata->varssorted, sepadata->nbilinvars);

   /* free the hashmap */
   SCIPhashmapFree(&sepadata->bilinvarsmap);

   sepadata->iscreated = FALSE;

   return SCIP_OKAY;
}

/** helper method to create separation data */
static
SCIP_RETCODE createSepaData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata            /**< separation data */
   )
{
   SCIP_CONS** conss;
   SCIP_VAR** vars;
   SCIP_CONSEXPR_ITERATOR* it;
   SCIP_HASHMAP* varmap;
   SCIP_VAR* x;
   SCIP_VAR* y;
   int xidx;
   int yidx;
   int i;
   int nconss;
   int nvars;
   int maxidx = 0;

   assert(sepadata != NULL);

   sepadata->nbilinvars = 0;
   sepadata->nbilinterms = 0;

   conss = SCIPconshdlrGetConss(sepadata->conshdlr);
   nconss = SCIPconshdlrGetNConss(sepadata->conshdlr);

   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   /* create variable map */
   SCIP_CALL( SCIPhashmapCreate(&varmap, SCIPblkmem(scip), SCIPgetNVars(scip)) );

   /* create iterator */
   SCIP_CALL( SCIPexpriteratorCreate(&it, sepadata->conshdlr, SCIPblkmem(scip)) );

   /* create the empty map for bilinear terms */
   SCIP_CALL( SCIPhashmapCreate(&sepadata->bilinvarsmap, SCIPblkmem(scip), nvars) );

   /* allocate memory for arrays */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sepadata->bilinauxvars, nvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sepadata->varssorted, nvars) );
   SCIP_CALL( SCIPallocClearBlockMemoryArray(scip, &sepadata->varpriorities, nvars) );

   /* find maximum variable index */
   for( i = 0; i < SCIPgetNVars(scip); ++i )
      maxidx = MAX(maxidx, SCIPvarGetIndex(vars[i]));  /*lint !e666*/
   sepadata->maxvarindex = maxidx;

   for( i = 0; i < nconss; ++i )
   {
      SCIP_CONSEXPR_EXPR* expr;

      SCIP_CALL(SCIPexpriteratorInit(it, SCIPgetExprConsExpr(scip, conss[i]), SCIP_CONSEXPRITERATOR_DFS, TRUE));
      SCIPexpriteratorSetStagesDFS(it, SCIP_CONSEXPRITERATOR_ENTEREXPR);
      expr = SCIPexpriteratorGetCurrent(it);

      /* collect bilinear terms */
      while( !SCIPexpriteratorIsEnd(it) ) /*lint !e441*/
      {
         switch( SCIPexpriteratorGetStageDFS(it) )
         {
            case SCIP_CONSEXPRITERATOR_ENTEREXPR:
            {
               SCIP_VAR* auxvar;
               int mapidx;
               int poslocks;
               int neglocks;

               assert(expr != NULL);

               auxvar = SCIPgetConsExprExprAuxVar(expr);

               /* no linearization variable available */
               if( auxvar == NULL )
                  break;

               x = NULL;
               y = NULL;

               /* test if expression is quadratic */
               if( SCIPgetConsExprExprHdlr(expr) == SCIPfindConsExprExprHdlr(sepadata->conshdlr, "pow")
                  && SCIPgetConsExprExprPowExponent(expr) == 2.0 )
               {
                  /* if only initial rows are requested, skip products of non-variable expressions */
                  if( sepadata->onlyinitial && !SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(expr)[0]) )
                     break;

                  x = SCIPgetConsExprExprAuxVar(SCIPgetConsExprExprChildren(expr)[0]);
                  y = x;
               }
                  /* test if expression is bilinear */
               else if( SCIPgetConsExprExprHdlr(expr) == SCIPfindConsExprExprHdlr(sepadata->conshdlr, "prod")
                  && SCIPgetConsExprExprNChildren(expr) == 2 )
               {
                  /* if only initial rows are requested, skip products of non-variable expressions */
                  if( sepadata->onlyinitial && (!SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(expr)[0])
                                                || !SCIPisConsExprExprVar(SCIPgetConsExprExprChildren(expr)[1])) )
                     break;

                  x = SCIPgetConsExprExprAuxVar(SCIPgetConsExprExprChildren(expr)[0]);
                  y = SCIPgetConsExprExprAuxVar(SCIPgetConsExprExprChildren(expr)[1]);
               }

               /* if children don't have linearization variables, there's nothing to do */
               if( x != NULL && y != NULL )
               {
                  /* switch variables if necessary */
                  if( SCIPvarComp(x, y) > 0 )
                     SCIPswapPointers((void**)&x, (void**)&y);

                  assert(auxvar != NULL);

                  xidx = SCIPvarGetIndex(x);
                  yidx = SCIPvarGetIndex(y);

                  /* compute unique index of the bilinear term */
                  mapidx = xidx * sepadata->maxvarindex + yidx;

                  if( !SCIPhashmapExists(sepadata->bilinvarsmap, (void*)(size_t) mapidx) )
                  {
                     /* store variables if its the first time they are found in a bilinear term */
                     if( !SCIPhashmapExists(varmap, (void*)(size_t) xidx) )
                     {
                        SCIP_CALL( SCIPhashmapInsertInt(varmap, (void*)(size_t) xidx, sepadata->nbilinvars) ); /*lint !e571*/
                        sepadata->varssorted[sepadata->nbilinvars] = x;
                        SCIP_CALL( SCIPcaptureVar(scip, x) );
                        ++sepadata->nbilinvars;
                     }

                     if( !SCIPhashmapExists(varmap, (void*)(size_t) yidx) )
                     {
                        SCIP_CALL( SCIPhashmapInsertInt(varmap, (void*)(size_t) yidx, sepadata->nbilinvars) ); /*lint !e571*/
                        sepadata->varssorted[sepadata->nbilinvars] = y;
                        SCIP_CALL( SCIPcaptureVar(scip, y) );
                        ++sepadata->nbilinvars;
                     }

                     /* insert linearization variable into auxvar hashmap */
                     SCIP_CALL( SCIPhashmapInsertInt(sepadata->bilinvarsmap, (void*)(size_t) mapidx,
                        sepadata->nbilinterms) ); /*lint !e571*/

                     /* add variables to bilin-arrays and capture them */
                     sepadata->bilinauxvars[sepadata->nbilinterms] = auxvar;
                     SCIP_CALL( SCIPcaptureVar(scip, auxvar) );
                     ++sepadata->nbilinterms;

                     /* add locks to priorities of both variables */
                     poslocks = SCIPgetConsExprExprNLocksPos(expr);
                     neglocks = SCIPgetConsExprExprNLocksNeg(expr);
                     sepadata->varpriorities[SCIPhashmapGetImageInt(varmap, (void*)(size_t) xidx)] += poslocks + neglocks; /*lint !e571*/
                     sepadata->varpriorities[SCIPhashmapGetImageInt(varmap, (void*)(size_t) yidx)] += poslocks + neglocks; /*lint !e571*/
                  }
                  else
                  {
                     expr = SCIPexpriteratorSkipDFS(it);
                     continue;
                  }
               }

               break;
            }

            default:
               SCIPABORT();
               break;
         }

         expr = SCIPexpriteratorGetNext(it);
      }
   }

   /* reallocate arrays to fit actually sizes */
   if( sepadata->nbilinvars < nvars )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &sepadata->varssorted, nvars, sepadata->nbilinvars) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &sepadata->varpriorities, nvars, sepadata->nbilinvars) );
   }

   if( sepadata->nbilinterms < nvars )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &sepadata->bilinauxvars, nvars, sepadata->nbilinterms) );
   }

   /* sort maxnumber of variables according to their occurrences */
   SCIPselectDownIntPtr(sepadata->varpriorities, (void**) sepadata->varssorted, sepadata->maxusedvars, sepadata->nbilinvars);

   SCIPexpriteratorFree(&it);
   SCIPhashmapFree(&varmap);

   sepadata->iscreated = TRUE;
   sepadata->isinitialround = TRUE;

   return SCIP_OKAY;
}

/** helper method to get the linearization variable of a bilinear term xy
 *
 *  @return NULL if no linearization variable exists
 */
static
SCIP_VAR* getBilinVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separation data */
   SCIP_VAR*             x,                  /**< first variable */
   SCIP_VAR*             y                   /**< second variable */
   )
{
   int idx;
   int img;

   assert(scip != NULL);
   assert(sepadata != NULL);
   assert(x != NULL);
   assert(y != NULL);

   /* it seems that x or y have been added after initsol -> no linearization variable available */
   if( SCIPvarGetIndex(x) > sepadata->maxvarindex || SCIPvarGetIndex(y) > sepadata->maxvarindex )
   {
      return NULL;
   }

   /* switch variables if necessary */
   if( x != y && SCIPvarComp(x, y) > 0 )
      SCIPswapPointers((void**) &x, (void**) &y);

   /* compute unique index of the bilinear term */
   idx = SCIPvarGetIndex(x) * sepadata->maxvarindex + SCIPvarGetIndex(y);

   if( SCIPhashmapExists(sepadata->bilinvarsmap, (void*)(size_t) idx) )
   {
      img = (int) SCIPhashmapGetImageInt(sepadata->bilinvarsmap, (void*)(size_t) idx); /*lint !e571*/
      return sepadata->bilinauxvars[img];
   }

   return NULL;
}

/** tests if a row contains too many unknown bilinear terms w.r.t. the parameters */
static
SCIP_RETCODE isAcceptableRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separation data */
   SCIP_ROW*             row,                /**< the row to be tested */
   SCIP_VAR*             var,                /**< the variable that is to be multiplied with row */
   int                   nlocks,             /**< the number of locks of the variable */
   SCIP_Bool*            acceptable          /**< buffer to store the result */
   )
{
   SCIP_VAR* linvar;
   int i;
   int nterms = 0;

   assert(row != NULL);
   assert(var != NULL);

   /* test if the ratio of non-zeroes and known terms of this variable is ok */
   if( SCIProwGetNNonz(row) * sepadata->maxnonzeroprop > nlocks )
   {
      *acceptable = FALSE;
      return SCIP_OKAY;
   }

   for( i = 0; (i < SCIProwGetNNonz(row)) && (sepadata->maxunknownterms >= 0 || nterms <= sepadata->maxunknownterms); ++i )
   {
      linvar = getBilinVar(scip, sepadata, var, SCIPcolGetVar(SCIProwGetCols(row)[i]) );

      if( linvar == NULL )
         ++nterms;
   }

   sepadata->currentnunknown = nterms;

   *acceptable = nterms <= sepadata->maxunknownterms;

   return SCIP_OKAY;
}

/** creates and returns rows of initial linear constraints */
static
SCIP_RETCODE getInitialRows(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW***           rows,               /**< buffer to store the rows */
   int*                  nrows               /**< buffer to store the number of linear rows */
   )
{
   SCIP_CONS** conss;
   SCIP_CONSHDLR* linhdlr;
   SCIP_CONSHDLR* knpsckhdlr;
   SCIP_CONSHDLR* varbndhdlr;
   SCIP_CONSHDLR* setppchdlr;
   int nconss;
   int i;

   assert(rows != NULL);
   assert(nrows != NULL);

   linhdlr = SCIPfindConshdlr(scip, "linear");
   knpsckhdlr = SCIPfindConshdlr(scip, "knapsack");
   varbndhdlr = SCIPfindConshdlr(scip, "varbound");
   setppchdlr = SCIPfindConshdlr(scip, "setppc");

   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);
   *nrows = 0;

   SCIP_CALL( SCIPallocBufferArray(scip, rows, nconss) );

   for( i = 0; i < nconss; ++i )
   {
      SCIP_ROW *row;

      if( SCIPconsGetHdlr(conss[i]) == linhdlr )
      {
         row = SCIPgetRowLinear(scip, conss[i]);
         SCIPdebugMsg(scip, "linear constraint found\n");
      }
      else if( SCIPconsGetHdlr(conss[i]) == knpsckhdlr )
      {
         row = SCIPgetRowKnapsack(scip, conss[i]);
         SCIPdebugMsg(scip, "knapsack constraint found\n");
      }
      else if( SCIPconsGetHdlr(conss[i]) == varbndhdlr )
      {
         row = SCIPgetRowVarbound(scip, conss[i]);
         SCIPdebugMsg(scip, "varbound constraint found\n");
      }
      else if( SCIPconsGetHdlr(conss[i]) == setppchdlr )
      {
         row = SCIPgetRowSetppc(scip, conss[i]);
         SCIPdebugMsg(scip, "setppc constraint found\n");
      }
      else
      {
         continue;
      }

      if( row != NULL)
      {
         (*rows)[*nrows] = row;
         ++*nrows;
      }
   }

   return SCIP_OKAY;
}

/** creates the RLT-cuts formed by multiplying a given row with (x - lb) or (ub - x)
 *
 * in detail:
 * -The row is multiplied either with (x - lb(x)) or with (ub(x) - x), depending on parameter uselb.
 * -The cut is computed either for lhs or rhs, depending on parameter uselhs.
 * -Terms for which no auxiliary variable exists are replaced by either McCormick, secants, or linearization cuts
 */
static
SCIP_RETCODE computeRltCuts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SEPADATA*        sepadata,           /**< separation data */
   SCIP_ROW**            cut,                /**< buffer to store the cut */
   SCIP_ROW*             row,                /**< the row that is used for the rlt cuts */
   SCIP_SOL*             sol,                /**< the point to be seperated (can be NULL) */
   SCIP_VAR*             var,                /**< the variable that is used for the rlt cuts */
   SCIP_Bool*            success,            /**< buffer to store whether cut was created successfully */
   SCIP_Bool             uselb,              /**< whether we multiply with (var - lb) or (ub - var) */
   SCIP_Bool             uselhs,             /**< whether to create a cut for the lhs or rhs */
   SCIP_Bool             local,              /**< whether local or global cuts should be computed */
   SCIP_Bool             computeEqCut        /**< whether conditions are fulfilled to compute equality cuts */
   )
{
   SCIP_Real signfactor;
   SCIP_Real boundfactor;
   SCIP_Real lbvar;
   SCIP_Real ubvar;
   SCIP_Real refpointvar;
   SCIP_Real coefvar;
   SCIP_Real constside;
   SCIP_Real finalside;
   int i;

   assert(sepadata != NULL);
   assert(cut != NULL);
   assert(row != NULL);
   assert(var != NULL);
   assert(success != NULL);
   assert(!computeEqCut || SCIPisEQ(scip, SCIProwGetLhs(row), SCIProwGetRhs(row)));

   *cut = NULL;

   /* get data for given variable */
   lbvar = local ? SCIPvarGetLbLocal(var) : SCIPvarGetLbGlobal(var);
   ubvar = local ? SCIPvarGetUbLocal(var) : SCIPvarGetUbGlobal(var);
   constside = uselhs ? SCIProwGetLhs(row) : SCIProwGetRhs(row);
   refpointvar = MAX(lbvar, MIN(ubvar, SCIPgetSolVal(scip, sol, var))); /*lint !e666*/

   /* if the bounds are too large or the respective side is infinity, skip this cut */
   if( REALABS(lbvar) > MAXVARBOUND || REALABS(ubvar) > MAXVARBOUND || SCIPisInfinity(scip, REALABS(constside)) )
   {
      SCIPdebugMsg(scip, "cut generation for row %s, %s and variable %s with its %s %f not possible\n",
         SCIProwGetName(row), uselhs ? "lhs" : "rhs", SCIPvarGetName(var),
         uselb ? "lower bound" : "upper bound", uselb ? lbvar : ubvar);

      *success = FALSE;
      return SCIP_OKAY;
   }

   /* initialize some factors needed for computation */
   coefvar = 0.0;
   finalside = 0.0;
   signfactor = (uselb ? 1.0 : -1.0);
   boundfactor = (uselb ? -lbvar : ubvar);

   *success = TRUE;

   /* create an empty row which we then fill with variables step by step */
   SCIP_CALL( SCIPcreateEmptyRowSepa(scip, cut, sepa, "rlt_cut", -SCIPinfinity(scip), SCIPinfinity(scip),
         TRUE, FALSE, FALSE) );

   /* iterate over all variables in the row and add the corresponding terms to the cuts */
   for( i = 0; i < SCIProwGetNNonz(row); ++i )
   {
      SCIP_VAR* auxvar;
      SCIP_VAR* colvar;
      SCIP_Real coefauxvar;
      SCIP_Real coefcolvar;

      colvar = SCIPcolGetVar(SCIProwGetCols(row)[i]);
      coefauxvar = SCIProwGetVals(row)[i] * signfactor;
      coefcolvar = SCIProwGetVals(row)[i] * boundfactor;

      auxvar = getBilinVar(scip, sepadata, var, colvar);

      /* if the auxiliary variable for this term exists, simply add it to the cut with the previous coefficient */
      if( auxvar != NULL )
      {
         SCIPdebugMsg(scip, "auxvar for %s found, will be added to cut\n", SCIPvarGetName(colvar));
         assert(!SCIPisInfinity(scip, coefauxvar));
         SCIP_CALL( SCIPaddVarToRow(scip, *cut, auxvar, coefauxvar) );
      }

      /* otherwise, use the McCormick estimator in place of the bilinear term */
      else if( colvar != var )
      {
         SCIP_Real lbcolvar = SCIPvarGetLbLocal(colvar);
         SCIP_Real ubcolvar = SCIPvarGetUbLocal(colvar);
         SCIP_Real refpointcolvar = MAX(lbcolvar, MIN(ubcolvar, SCIPgetSolVal(scip, sol, colvar))); /*lint !e666*/

         assert(!computeEqCut);

         if( REALABS(lbcolvar) > MAXVARBOUND || REALABS(ubcolvar) > MAXVARBOUND )
         {
            *success = FALSE;
            return SCIP_OKAY;
         }

         SCIPdebugMsg(scip, "auxvar for %s not found, will use McCormick estimators\n", SCIPvarGetName(colvar));

         SCIPaddBilinMcCormick(scip, coefauxvar, lbvar, ubvar, refpointvar, lbcolvar,
            ubcolvar, refpointcolvar, uselhs, &coefvar, &coefcolvar, &finalside, success);

         if( !*success )
            return SCIP_OKAY;
      }

      /* or, if it's a quadratic term, use a secant for overestimation and a gradient for underestimation */
      else
      {
         SCIPdebugMsg(scip, "auxvar for %s not found, will use gradient and secant estimators\n", SCIPvarGetName(colvar));

         assert(!computeEqCut);

         /* depending on over-/underestimation and the sign of the column variable, compute secant or tangent */
         if( (uselhs && coefauxvar > 0.0) || (!uselhs && coefauxvar < 0.0) )
            SCIPaddSquareSecant(scip, coefauxvar, lbvar, ubvar, refpointvar, &coefvar, &finalside, success);
         else
            SCIPaddSquareLinearization(scip, coefauxvar, refpointvar, SCIPvarIsIntegral(var), &coefvar, &finalside, success);

         if( !*success )
            return SCIP_OKAY;
      }

      /* add the linear term for this column */
      if( colvar != var )
      {
         assert(!SCIPisInfinity(scip, coefcolvar));
         SCIP_CALL( SCIPaddVarToRow(scip, *cut, colvar, coefcolvar) );
      }
      else
         coefvar += coefcolvar;
   }

   if( REALABS(finalside) > MAXVARBOUND )
   {
      *success = FALSE;
      return SCIP_OKAY;
   }

   /* multiply (x-lb) or (ub -x) with the lhs and rhs of the row */
   coefvar += signfactor * (SCIProwGetConstant(row) - constside);
   finalside = boundfactor * (constside - SCIProwGetConstant(row)) - finalside;

   /* set the coefficient of var and the constant side */
   assert(!SCIPisInfinity(scip, coefvar));
   SCIP_CALL( SCIPaddVarToRow(scip, *cut, var, coefvar) );

   assert(!SCIPisInfinity(scip, finalside));
   if( uselhs || computeEqCut )
   {
      SCIP_CALL( SCIPchgRowLhs(scip, *cut, finalside) );
   }
   if( !uselhs || computeEqCut )
   {
      SCIP_CALL( SCIPchgRowRhs(scip, *cut, finalside) );
   }

   SCIPdebugMsg(scip, "cut was generated successfully:\n");
#ifdef SCIP_DEBUG
   SCIP_CALL( SCIPprintRow(scip, *cut, NULL) );
#endif

   return SCIP_OKAY;
}

/*
 * Callback methods of separator
 */

/** copy method for separator plugins (called when SCIP copies plugins) */
static
SCIP_DECL_SEPACOPY(sepaCopyRlt)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(sepa != NULL);
   assert(strcmp(SCIPsepaGetName(sepa), SEPA_NAME) == 0);

   /* call inclusion method of separator */
   SCIP_CALL( SCIPincludeSepaRlt(scip) );

   return SCIP_OKAY;
}

/** destructor of separator to free user data (called when SCIP is exiting) */
static
SCIP_DECL_SEPAFREE(sepaFreeRlt)
{  /*lint --e{715}*/
   SCIP_SEPADATA* sepadata;

   assert(strcmp(SCIPsepaGetName(sepa), SEPA_NAME) == 0);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   /* free separator data */
   SCIPfreeBlockMemory(scip, &sepadata);

   SCIPsepaSetData(sepa, NULL);

   return SCIP_OKAY;
}

/** solving process deinitialization method of separator (called before branch and bound process data is freed) */
static
SCIP_DECL_SEPAEXITSOL(sepaExitsolRlt)
{  /*lint --e{715}*/
   SCIP_SEPADATA* sepadata;

   assert(strcmp(SCIPsepaGetName(sepa), SEPA_NAME) == 0);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   if( sepadata->iscreated )
   {
      SCIP_CALL( freeSepaData(scip, sepadata) );
   }

   return SCIP_OKAY;
}

/** LP solution separation method of separator */
static
SCIP_DECL_SEPAEXECLP(sepaExeclpRlt)
{  /*lint --e{715}*/
   SCIP_ROW** rows;
   SCIP_SEPADATA* sepadata;
   int ncalls;
   int depth;
   int ncuts;
   int nrows;
   int i;
   int j;
   int k;

   assert(strcmp(SCIPsepaGetName(sepa), SEPA_NAME) == 0);

   SCIPdebugMsg(scip, "separator called\n");

   sepadata = SCIPsepaGetData(sepa);

   *result = SCIP_DIDNOTRUN;

   if( sepadata->maxncuts == 0 )
   {
      SCIPdebugMsg(scip, "exit seperator because maxncuts is set to 0\n");
      return SCIP_OKAY;
   }

   /* don't run in a sub-SCIP or in probing */
   if( SCIPgetSubscipDepth(scip) > 0 && !sepadata->useinsubscip )
   {
      SCIPdebugMsg(scip, "exit seperator because in sub-SCIP\n");
      return SCIP_OKAY;
   }

   /* don't run in a sub-SCIP or in probing */
   if( SCIPinProbing(scip) )
   {
      SCIPdebugMsg(scip, "exit seperator because in or probing\n");
      return SCIP_OKAY;
   }

   /* only call separator a given number of times at each node */
   depth = SCIPgetDepth(scip);
   ncalls = SCIPsepaGetNCallsAtNode(sepa);
   if( (depth == 0 && sepadata->maxroundsroot >= 0 && ncalls >= sepadata->maxroundsroot)
        || (depth > 0 && sepadata->maxrounds >= 0 && ncalls >= sepadata->maxrounds) )
   {
      SCIPdebugMsg(scip, "exit seperator because round limit for this node is reached\n");
      return SCIP_OKAY;
   }

   /* if this is called for the first time, create the sepadata and start the initial separation round */
   if( !sepadata->iscreated )
   {
      *result = SCIP_DIDNOTFIND;
      SCIP_CALL( createSepaData(scip, sepadata) );
   }

   /* no bilinear terms available -> skip */
   if( sepadata->nbilinvars == 0 )
   {
      SCIPdebugMsg(scip, "exit seperator because there are no known bilinear terms\n");
      return SCIP_OKAY;
   }

   /* only call separator, if we are not close to terminating */
   if( SCIPisStopped(scip) )
   {
      SCIPdebugMsg(scip, "exit seperator because we are too close to terminating\n");
      return SCIP_OKAY;
   }

   /* only call separator, if an optimal LP solution is at hand */
   if( SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
   {
      SCIPdebugMsg(scip, "exit seperator because there is no LP solution at hand\n");
      return SCIP_OKAY;
   }

   /* get the rows, depending on settings */
   if( sepadata->isinitialround || sepadata->onlyinitial )
   {
      SCIP_CALL( getInitialRows(scip, &rows, &nrows) );
   }
   else
   {
      SCIP_CALL( SCIPgetLPRowsData(scip, &rows, &nrows) );
   }

   for( i = 0; i < nrows && !SCIPisStopped(scip); ++i )
   {
      SCIP_Bool iseqrow = SCIPisEQ(scip, SCIProwGetLhs(rows[i]), SCIProwGetRhs(rows[i]));

      /* if equality rows are requested, only those can be used */
      if( sepadata->onlyeqrows && !iseqrow )
         continue;

      /* if global cuts are requested, only globally valid rows can be used */
      if( !allowlocal && SCIProwIsLocal(rows[i]))
         continue;

      /* if continuous rows are requested, only those can be used */
      if( sepadata->onlycontrows )
      {
         SCIP_COL** cols = SCIProwGetCols(rows[i]);
         SCIP_Bool iscontrow = TRUE;

         /* check row for integral variables */
         for( j = 0; j < SCIProwGetNNonz(rows[i]); ++j )
         {
            if( SCIPcolIsIntegral(cols[j]) )
            {
               iscontrow = FALSE;
               break;
            }
         }

         if( !iscontrow )
            continue;
      }

      /* don't try to use rows that have been generated by the RLT separator
       *
       * @TODO check whether name for McCormick cuts changes
       */
      if( SCIProwGetOriginSepa(rows[i]) == sepa || strcmp(SCIProwGetName(rows[i]), "mccormick") == 0 )
         continue;

      ncuts = 0;
      *result = SCIP_DIDNOTFIND;

      for( j = 0; j < sepadata->nbilinvars && (sepadata->maxusedvars < 0 || j < sepadata->maxusedvars); ++j )
      {
         SCIP_VAR *var = sepadata->varssorted[j];
         SCIP_Bool uselb[4] = {TRUE, TRUE, FALSE, FALSE};
         SCIP_Bool uselhs[4] = {TRUE, FALSE, TRUE, FALSE};
         SCIP_Bool buildeqcut;
         SCIP_Bool accepted;
         SCIP_Bool success;
         SCIP_ROW *cut;

         /* check whether this row and var fulfill the conditions */
         SCIP_CALL(isAcceptableRow(scip, sepadata, rows[i], var, sepadata->varpriorities[j], &accepted));

         if( !accepted )
         {
            SCIPdebugMsg(scip, "rejected row %s for variable %s\n", SCIProwGetName(rows[i]), SCIPvarGetName(var));
            continue;
         }

         SCIPdebugMsg(scip, "accepted row %s for variable %s\n", SCIProwGetName(rows[i]), SCIPvarGetName(var));
#ifdef SCIP_DEBUG
         SCIP_CALL( SCIPprintRow(scip, rows[i], NULL) );
#endif

         /* if all terms are known and it is an equality row, compute equality cuts */
         buildeqcut = (sepadata->currentnunknown == 0 && iseqrow);

         /* go over all combinations of sides and bounds and compute the respective cuts */
         for( k = 0; k < 4; ++k )
         {
            /* if equality cuts are possible, lhs and rhs cuts are equal so skip rhs */
            if( buildeqcut && (k % 2 == 1) )
               continue;

            success = TRUE;

            SCIPdebugMsg(scip, "starting cut generation for row %s, %s and variable %s with its %s %s\n",
               SCIProwGetName(rows[i]), uselhs[k] ? "lhs" : "rhs", SCIPvarGetName(var),
               allowlocal ? "local" : "global",
               uselb[k] ? "lower bound" : "upper bound");

            /* compute the rlt cut */
            SCIP_CALL(computeRltCuts(scip, sepa, sepadata, &cut, rows[i], NULL, var, &success, uselb[k], uselhs[k],
               allowlocal, buildeqcut));

            SCIPdebugMsg(scip, "finished cut generation for row %s, %s and variable %s with its %s %s\n",
               SCIProwGetName(rows[i]), uselhs[k] ? "lhs" : "rhs", SCIPvarGetName(var),
               allowlocal ? "local" : "global",
               uselb[k] ? "lower bound" : "upper bound");

            /* if the cut was created successfully and is violated, it is added to SCIP */
            if( success )
            {
               if( SCIPisFeasLT(scip, SCIPgetRowFeasibility(scip, cut), 0.0) )
               {
                  SCIP_Bool infeasible;

                  /* add the row to SCIP; equality cuts are forced to be added to the LP */
                  SCIP_CALL(SCIPaddRow(scip, cut, buildeqcut, &infeasible));
                  ++ncuts;

                  if( infeasible )
                  {
                     SCIPdebugMsg(scip, "CUTOFF! At least one of the cuts revealed infeasibility!\n");
                     *result = SCIP_CUTOFF;
                  } else
                  {
                     SCIPdebugMsg(scip, "SEPARATED: added cut to scip\n");
                     *result = SCIP_SEPARATED;
                  }
               }
               else
                  SCIPdebugMsg(scip, "the cut was created successfully, but not accepted by scip\n");
            } else
               SCIPdebugMsg(scip, "the generation of the cut failed\n");

            /* release the cut */
            if( cut != NULL)
            {
               SCIP_CALL(SCIPreleaseRow(scip, &cut));
            }

            if( (sepadata->maxncuts >= 0 && ncuts >= sepadata->maxncuts) || *result == SCIP_CUTOFF )
            {
               SCIPdebugMsg(scip, "exit seperator because we found enough cuts or a cutoff -> skip\n");

               if( sepadata->isinitialround || sepadata->onlyinitial )
               {
                  SCIPfreeBufferArray(scip, &rows);
                  sepadata->isinitialround = FALSE;
               }
               return SCIP_OKAY;
            }
         }
      }
   }

   SCIPdebugMsg(scip, "exit seperator because cut calculation is finished\n");

   if( sepadata->isinitialround || sepadata->onlyinitial )
   {
      SCIPfreeBufferArray(scip, &rows);
      sepadata->isinitialround = FALSE;
   }

   return SCIP_OKAY;
}

/*
 * separator specific interface methods
 */

/** creates the RLT separator and includes it in SCIP */
SCIP_RETCODE SCIPincludeSepaRlt(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_SEPADATA* sepadata;
   SCIP_SEPA* sepa;

   /* create RLT separator data */
   SCIP_CALL( SCIPallocClearBlockMemory(scip, &sepadata) );
   sepadata->conshdlr = SCIPfindConshdlr(scip, "expr");

   /* include separator */
   SCIP_CALL( SCIPincludeSepaBasic(scip, &sepa, SEPA_NAME, SEPA_DESC, SEPA_PRIORITY, SEPA_FREQ, SEPA_MAXBOUNDDIST,
         SEPA_USESSUBSCIP, SEPA_DELAY, sepaExeclpRlt, NULL, sepadata) );

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetSepaCopy(scip, sepa, sepaCopyRlt) );
   SCIP_CALL( SCIPsetSepaFree(scip, sepa, sepaFreeRlt) );
   SCIP_CALL( SCIPsetSepaExitsol(scip, sepa, sepaExitsolRlt) );

   /* add RLT separator parameters */
   SCIP_CALL( SCIPaddIntParam(scip,
         "separating/" SEPA_NAME "/maxncuts",
         "maximal number of rlt-cuts that are added per round (-1: unlimited)",
         &sepadata->maxncuts, FALSE, DEFAULT_MAXNCUTS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
         "separating/" SEPA_NAME "/maxunknownterms",
         "maximal number of unknown bilinear terms a row is still used with (-1: unlimited)",
         &sepadata->maxunknownterms, FALSE, DEFAULT_MAXUNKNOWNTERMS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
         "separating/" SEPA_NAME "/maxusedvars",
         "maximal number of variables used to compute rlt cuts (-1: unlimited)",
         &sepadata->maxusedvars, FALSE, DEFAULT_MAXUSEDVARS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "separating/" SEPA_NAME "/maxnonzeroprop",
         "maximal proportion of known bilinear terms of a variable to non-zeroes of a row that is adccepted",
         &sepadata->maxnonzeroprop, FALSE, DEFAULT_MAXNONZEROPROP, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
      "separating/" SEPA_NAME "/maxrounds",
      "maximal number of eccuts separation rounds per node (-1: unlimited)",
      &sepadata->maxrounds, FALSE, DEFAULT_MAXROUNDS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
      "separating/" SEPA_NAME "/maxroundsroot",
      "maximal number of eccuts separation rounds in the root node (-1: unlimited)",
      &sepadata->maxroundsroot, FALSE, DEFAULT_MAXROUNDSROOT, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "separating/" SEPA_NAME "/onlyeqrows",
      "if set to true, only equality rows are used for rlt cuts",
      &sepadata->onlyeqrows, FALSE, DEFAULT_ONLYEQROWS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "separating/" SEPA_NAME "/onlycontrows",
      "if set to true, only continuous rows are used for rlt cuts",
      &sepadata->onlycontrows, FALSE, DEFAULT_ONLYCONTROWS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "separating/" SEPA_NAME "/onlyinitial",
      "if set to true, only initial constraints are used",
      &sepadata->onlyinitial, FALSE, DEFAULT_ONLYINITIAL, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "separating/" SEPA_NAME "/useinsubscip",
      "if set to true, rlt is also used in sub-scips",
      &sepadata->useinsubscip, FALSE, DEFAULT_USEINSUBSCIP, NULL, NULL) );

   return SCIP_OKAY;
}
