/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2020 Konrad-Zuse-Zentrum                            */
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
 * @author Ksenia Bestuzheva
 *
 * @todo implement the possibility to add extra auxiliary variables for RLT (like in DOI 10.1080/10556788.2014.916287)
 * @todo add RLT cuts for the product of equality constraints
 * @todo implement dynamic addition of RLT cuts during branching (see DOI 10.1007/s10898-012-9874-7)
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/set.h"
#include "scip/sepa_rlt.h"
#include "scip/cons_expr.h"
#include "scip/cons_linear.h"
#include "scip/cons_knapsack.h"
#include "scip/cons_varbound.h"
#include "scip/cons_setppc.h"
#include "scip/struct_scip.h"


#define SEPA_NAME              "rlt"
#define SEPA_DESC              "rlt separator"
#define SEPA_PRIORITY                10 /**< priority for separation */
#define SEPA_FREQ                     0 /**< frequency for separating cuts; zero means to separate only in the root node */
#define SEPA_MAXBOUNDDIST           1.0 /**< maximal relative distance from the current node's dual bound to primal bound
+                                        *   compared to best node's dual bound for applying separation.*/
#define SEPA_USESSUBSCIP          FALSE /**< does the separator use a secondary SCIP instance? */
#define SEPA_DELAY                FALSE /**< should separation method be delayed, if other separators found cuts? */

#define DEFAULT_MAXUNKNOWNTERMS      -1 /**< default value for parameter maxunknownterms */
#define DEFAULT_MAXUSEDVARS         100 /**< default value for parameter maxusedvars */
#define DEFAULT_MAXNCUTS             -1 /**< default value for parameter maxncuts */
#define DEFAULT_MAXROUNDS             1 /**< default value for parameter maxrounds */
#define DEFAULT_MAXROUNDSROOT        10 /**< default value for parameter maxroundsroot */
#define DEFAULT_ONLYEQROWS        FALSE /**< default value for parameter eqrowsfirst */
#define DEFAULT_ONLYCONTROWS      FALSE /**< default value for parameter eqrowsfirst */
#define DEFAULT_ONLYINITIAL        TRUE /**< default value for parameter onlyinitial */
#define DEFAULT_USEINSUBSCIP      FALSE /**< default value for parameter useinsubscip */
#define DEFAULT_USEPROJECTION     FALSE /**< default value for parameter useprojection */
#define DEFAULT_DETECTHIDDEN      FALSE /**< default value for parameter detecthidden */
#define DEFAULT_HIDDENRLT         FALSE /**< default value for parameter hiddenrlt */
#define DEFAULT_ADDTOPOOL          TRUE /**< default value for parameter addtopool */

#define DEFAULT_GOODSCORE           1.0 /**< threshold for score of cut relative to best score to be considered good,
                                         *   so that less strict filtering is applied */
#define DEFAULT_BADSCORE            0.5 /**< threshold for score of cut relative to best score to be discarded */
#define DEFAULT_OBJPARALWEIGHT      0.0 /**< weight of objective parallelism in cut score calculation */
#define DEFAULT_EFFICACYWEIGHT      1.0 /**< weight of efficacy in cut score calculation */
#define DEFAULT_DIRCUTOFFDISTWEIGHT 0.0 /**< weight of directed cutoff distance in cut score calculation */
#define DEFAULT_GOODMAXPARALL       0.1 /**< maximum parallelism for good cuts */
#define DEFAULT_MAXPARALL           0.1 /**< maximum parallelism for non-good cuts */

#define MAXVARBOUND                1e+5 /**< maximum allowed variable bound for computing an RLT-cut */

/*
 * Data structures
 */

/** data object for pairs and triples of variables */
struct HashData
{
   SCIP_VAR**            vars;               /**< variables in the pair or triple, used for hash comparison */
   int                   nvars;              /**< number of variables */
   int                   nrows;              /**< number of rows */
   int                   firstrow;           /**< beginning of the corresponding row linked list */
};
typedef struct HashData HASHDATA;

/** data of a bilinear (i.e. appearing in bilinear products) variable */
struct BilinVarData
{
   int                   priority;           /**< priority of the variable */
   SCIP_VAR**            varbilinvars;       /**< vars appearing in a bilinear term together with the variable */
   int                   nvarbilinvars;      /**< number of vars in varbilinvars */
   int                   svarbilinvars;      /**< size of varbilinvars */
};
typedef struct BilinVarData BILINVARDATA;

/** separator data */
struct SCIP_SepaData
{
   SCIP_CONSHDLR*        conshdlr;           /**< expression constraint handler */
   SCIP_Bool             iscreated;          /**< indicates whether the sepadata has been initialized yet */
   SCIP_Bool             isinitialround;     /**< indicates that this is the first round and initial rows are used */

   /* bilinear variables */
   SCIP_HASHMAP*         bilinvarsmap;       /**< map for accessing the linearization variables/exprs of each bilinear term */
   SCIP_VAR**            varssorted;         /**< variables that occur in bilinear terms sorted by priority */
   BILINVARDATA**        bilinvardatas;      /**< for each bilinear var: all vars that appear together with it in a product */
   int                   nbilinvars;         /**< total number of variables occurring in bilinear terms */
   int                   sbilinvars;         /**< size of arrays for variables occurring in bilinear terms */

   /* information on linearisations of bilinear products */
   int*                  eqlinexpr;          /**< position of the linexpr that is equal to the product (nlinexprs[i] if none) */
   int                   nbilinterms;        /**< total number of bilinear terms */

   /* parameters */
   int                   maxunknownterms;    /**< maximum number of unknown bilinear terms a row can have to be used */
   int                   maxusedvars;        /**< maximum number of variables that will be used to compute rlt cuts */
   int                   maxncuts;           /**< maximum number of cuts that will be added per round */
   int                   maxrounds;          /**< maximal number of separation rounds per node (-1: unlimited) */
   int                   maxroundsroot;      /**< maximal number of separation rounds in the root node (-1: unlimited) */
   SCIP_Bool             onlyeqrows;         /**< whether only equality rows should be used for rlt cuts */
   SCIP_Bool             onlycontrows;       /**< whether only continuous rows should be used for rlt cuts */
   SCIP_Bool             onlyinitial;        /**< whether only initial rows should be used for rlt cuts */
   SCIP_Bool             useinsubscip;       /**< whether the separator should also be used in sub-scips */
   SCIP_Bool             useprojection;      /**< whether the separator should first check projected rows */
   SCIP_Bool             detecthidden;       /**< whether implicit products should be detected and separated by McCormick */
   SCIP_Bool             hiddenrlt;          /**< whether RLT cuts (TRUE) or only McCormick inequalities (FALSE) should be added for hidden products */
   SCIP_Bool             addtopool;          /**< whether globally valid RLT cuts are added to the global cut pool */

   /* cut selection parameters */
   SCIP_Real             goodscore;          /**< threshold for score of cut relative to best score to be considered good,
                                              *   so that less strict filtering is applied */
   SCIP_Real             badscore;           /**< threshold for score of cut relative to best score to be discarded */
   SCIP_Real             objparalweight;     /**< weight of objective parallelism in cut score calculation */
   SCIP_Real             efficacyweight;     /**< weight of efficacy in cut score calculation */
   SCIP_Real             dircutoffdistweight;/**< weight of directed cutoff distance in cut score calculation */
   SCIP_Real             goodmaxparall;      /**< maximum parallelism for good cuts */
   SCIP_Real             maxparall;          /**< maximum parallelism for non-good cuts */

   /* TODO remove this when done with cliques */
   SCIP_CLOCK*           cliquetime;         /**< time spent on handling cliques in detection */
};

/* a simplified representation of an LP row */
struct RLT_SimpleRow
{
   SCIP_Real*            coefs;              /**< coefficients */
   SCIP_VAR**            vars;               /**< variables */
   SCIP_Real             rhs;                /**< right hand side */
   SCIP_Real             lhs;                /**< left hand side */
   SCIP_Real             cst;                /**< constant */
   int                   nNonz;              /**< number of nonzeroes */
   int                   size;               /**< size of the coefs and vars arrays */
};
typedef struct RLT_SimpleRow RLT_SIMPLEROW;

/*
 * Local methods
 */

/** returns TRUE iff both keys are equal; two constraint arrays are equal if they have the same pointer */
static
SCIP_DECL_HASHKEYEQ(hashdataKeyEqConss)
{  /*lint --e{715}*/
   HASHDATA* hashdata1;
   HASHDATA* hashdata2;
   int v;

   hashdata1 = (HASHDATA*)key1;
   hashdata2 = (HASHDATA*)key2;

   /* check data structure */
   assert(hashdata1->nvars == hashdata2->nvars);
   assert(hashdata1->firstrow != -1 || hashdata2->firstrow != -1);

   for( v = hashdata1->nvars-1; v >= 0; --v )
   {
      /* tests if variables are equal */
      if( hashdata1->vars[v] != hashdata2->vars[v] )
         return FALSE;

      assert(SCIPvarCompare(hashdata1->vars[v], hashdata2->vars[v]) == 0);
   }

   /* a hashdata object is only equal if it has the same constraint array pointer */
   if( hashdata1->firstrow == -1 || hashdata2->firstrow == -1 || hashdata1->firstrow == hashdata2->firstrow )
      return TRUE;
   else
      return FALSE;
}

/** returns the hash value of the key */
static
SCIP_DECL_HASHKEYVAL(hashdataKeyValConss)
{  /*lint --e{715}*/
   HASHDATA* hashdata;
   int minidx;
   int mididx;
   int maxidx;
   int idx[3];

   hashdata = (HASHDATA*)key;
   assert(hashdata != NULL);
   assert(hashdata->vars != NULL);
   assert(hashdata->nvars == 3 || hashdata->nvars == 2);

   idx[0] = SCIPvarGetIndex(hashdata->vars[0]);
   idx[1] = SCIPvarGetIndex(hashdata->vars[1]);
   idx[2] = SCIPvarGetIndex(hashdata->vars[hashdata->nvars - 1]);

   minidx = MIN(idx[0], MIN(idx[1], idx[2]));
   maxidx = MAX(idx[0], MAX(idx[1], idx[2]));
   if( idx[0] == maxidx )
      mididx = MAX(idx[1], idx[2]);
   else
      mididx = MAX(idx[0], MIN(idx[1], idx[2]));

   /* vars should already be sorted by index */
   assert(minidx <= mididx && mididx <= maxidx);

   return SCIPhashFour(hashdata->nvars, minidx, mididx, maxidx);
}


/* helper method to free the separation data */
static
SCIP_RETCODE freeSepaData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata            /**< separation data */
   )
{  /*lint --e{715}*/
   int i;
   BILINVARDATA* bilinvardata;

   assert(sepadata->iscreated);
   assert(sepadata->bilinvarsmap != NULL);

   if( sepadata->nbilinvars != 0 )
   {
      /* release bilinvars that were captured for rlt and free all related arrays */

      for( i = 0; i < sepadata->nbilinvars; ++i )
      {
         assert(sepadata->varssorted[i] != NULL);
         SCIP_CALL( SCIPreleaseVar(scip, &(sepadata->varssorted[i])) );
         bilinvardata = sepadata->bilinvardatas[i];
         SCIPfreeBlockMemoryArray(scip, &bilinvardata->varbilinvars, bilinvardata->svarbilinvars);
         SCIPfreeBlockMemory(scip, &bilinvardata);
      }
      SCIPfreeBlockMemoryArray(scip, &sepadata->bilinvardatas, sepadata->sbilinvars);
      SCIPfreeBlockMemoryArray(scip, &sepadata->varssorted, sepadata->sbilinvars);
      sepadata->nbilinvars = 0;
   }

   /* free the remaining array */
   if( sepadata->nbilinterms > 0 )
   {
      SCIPfreeBlockMemoryArray(scip, &sepadata->eqlinexpr, sepadata->nbilinterms);
   }

   /* free the hashmap */
   SCIPhashmapFree(&sepadata->bilinvarsmap);

   sepadata->iscreated = FALSE;

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

      if( row != NULL )
      {
         (*rows)[*nrows] = row;
         ++*nrows;
      }
   }

   return SCIP_OKAY;
}

/* make sure that the arrays in sepadata are large enough to store information on n variables */
static
SCIP_RETCODE ensureVarsSize(
   SCIP*                 scip,
   SCIP_SEPADATA*        sepadata,
   int                   n
   )
{
   int newsize;

   /* check whether array is large enough */
   if( n <= sepadata->sbilinvars )
      return SCIP_OKAY;

   /* compute new size */
   newsize = SCIPcalcMemGrowSize(scip, n);
   assert(n <= newsize);

   /* realloc arrays */
   SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &sepadata->varssorted, sepadata->sbilinvars, newsize) );
   SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &sepadata->bilinvardatas, sepadata->sbilinvars, newsize) );

   sepadata->sbilinvars = newsize;

   return SCIP_OKAY;
}

/** compares the priority of two bilinear variables, returns -1 if first is smaller than, and +1 if first is greater
* than second variable priority; returns 0 if both priorities are equal
*/
static
int bilinVarDataCompare(
   BILINVARDATA*         bilinvardata1,               /**< data of the first bilinear variable */
   BILINVARDATA*         bilinvardata2                /**< data of the second bilinear variable */
)
{
   assert(bilinvardata1 != NULL);
   assert(bilinvardata2 != NULL);

   if( bilinvardata1->priority < bilinvardata2->priority )
      return -1;
   else if( bilinvardata1->priority > bilinvardata2->priority )
      return +1;
   else
      return 0;
}

/** comparison method for sorting bilinear variable data by non-decreasing priority */
static
SCIP_DECL_SORTPTRCOMP(bilinVarDataComp)
{
   return bilinVarDataCompare((BILINVARDATA*)elem1, (BILINVARDATA*)elem2);
}

/* make sure that arrays in bilinvardata in sepadata is large enough to store n variables */
static
SCIP_RETCODE ensureBilinVarDataSize(
   SCIP*                 scip,
   BILINVARDATA*         bilinvardata,
   int                   n
   )
{
   int newsize;

   /* check whether array is large enough */
   if( n <= bilinvardata->svarbilinvars )
      return SCIP_OKAY;

   /* compute new size */
   newsize = SCIPcalcMemGrowSize(scip, n);
   assert(n <= newsize);

   /* realloc array */
   SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &bilinvardata->varbilinvars, bilinvardata->svarbilinvars, newsize) );

   bilinvardata->svarbilinvars = newsize;

   return SCIP_OKAY;
}

/** saves variables x and y to separator data and stores information about their connection
 *
 *  variables must be captured separately
 */
static
SCIP_RETCODE addProductVars(
  SCIP*                  scip,               /**< SCIP data structure */
  SCIP_SEPADATA*         sepadata,           /**< separator data */
  SCIP_VAR*              x,                  /**< x variable */
  SCIP_VAR*              y,                  /**< y variable */
  SCIP_HASHMAP*          varmap,             /**< hashmap linking var index to position */
  int                    nlocks              /**< number of locks */
  )
{
   int xpos;
   int ypos;
   int xidx;
   int yidx;
   int pos;
   int i;
   SCIP_Bool found;
   BILINVARDATA* xdata;
   BILINVARDATA* ydata;

   xidx = SCIPvarGetIndex(x);
   yidx = SCIPvarGetIndex(y);

   xpos = SCIPhashmapGetImageInt(varmap, (void*)(size_t) xidx); /*lint !e571 */

   if( xpos == INT_MAX )
   {
      SCIP_CALL( SCIPhashmapInsertInt(varmap, (void*)(size_t) xidx, sepadata->nbilinvars) ); /*lint !e571*/
      SCIP_CALL( ensureVarsSize(scip, sepadata, sepadata->nbilinvars + 1) );
      sepadata->varssorted[sepadata->nbilinvars] = x;
      SCIP_CALL( SCIPallocClearBlockMemory(scip, &sepadata->bilinvardatas[sepadata->nbilinvars]) ); /*lint !e866*/
      xpos = sepadata->nbilinvars;
      ++sepadata->nbilinvars;
   }

   assert(sepadata->bilinvardatas != NULL);
   assert(xpos >= 0 && xpos < sepadata->nbilinvars );

   xdata = sepadata->bilinvardatas[xpos];
   if( xdata->varbilinvars == NULL )
   {
      found = FALSE;
      pos = 0;
   }
   else
   {
      found = SCIPsortedvecFindPtr((void**) xdata->varbilinvars, SCIPvarComp, y, xdata->nvarbilinvars, &pos);
   }

   if( !found )
   {
      SCIP_CALL( ensureBilinVarDataSize(scip, xdata, xdata->nvarbilinvars + 1) );
      for( i = xdata->nvarbilinvars; i > pos; --i )
      {
         xdata->varbilinvars[i] = xdata->varbilinvars[i - 1];
      }
      xdata->varbilinvars[pos] = y;
      ++xdata->nvarbilinvars;
   }

   ypos = SCIPhashmapGetImageInt(varmap, (void*)(size_t) yidx); /*lint !e571 */

   if( ypos == INT_MAX )
   {
      SCIP_CALL( SCIPhashmapInsertInt(varmap, (void*)(size_t) yidx, sepadata->nbilinvars) ); /*lint !e571*/
      SCIP_CALL( ensureVarsSize(scip, sepadata, sepadata->nbilinvars + 1) );
      sepadata->varssorted[sepadata->nbilinvars] = y;
      SCIP_CALL( SCIPallocClearBlockMemory(scip, &sepadata->bilinvardatas[sepadata->nbilinvars]) ); /*lint !e866*/
      ypos = sepadata->nbilinvars;
      ++sepadata->nbilinvars;
   }

   assert(ypos >= 0 && ypos < sepadata->nbilinvars );

   ydata = sepadata->bilinvardatas[ypos];
   if( xidx != yidx )
   {
      if( ydata->varbilinvars == NULL )
      {
         found = FALSE;
         pos = 0;
      }
      else
      {
         found = SCIPsortedvecFindPtr((void**) ydata->varbilinvars, SCIPvarComp, x, ydata->nvarbilinvars, &pos);
      }

      if( !found )
      {
         SCIP_CALL( ensureBilinVarDataSize(scip, ydata, ydata->nvarbilinvars + 1) );
         for( i = ydata->nvarbilinvars; i > pos; --i )
         {
            ydata->varbilinvars[i] = ydata->varbilinvars[i - 1];
         }
         ydata->varbilinvars[pos] = x;
         ++ydata->nvarbilinvars;
      }
   }

   assert(xpos == SCIPhashmapGetImageInt(varmap, (void*)(size_t) xidx)); /*lint !e571 */
   assert(ypos == SCIPhashmapGetImageInt(varmap, (void*)(size_t) yidx)); /*lint !e571 */

   /* add locks to priorities of both variables */
   xdata->priority += nlocks;
   ydata->priority += nlocks;

   return SCIP_OKAY;
}

/** extract a bilinear product from two linear relations, if possible */
static
SCIP_RETCODE extractProducts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_VAR**            vars,               /**< 3 variables involved in the inequalities in the order x,w,y */
   SCIP_Real*            coefs1,             /**< coefficients of the first inequality */
   SCIP_Real*            coefs2,             /**< coefficients of the second inequality */
   SCIP_Real             side1,              /**< side of the first (implied) inequality */
   SCIP_Real             side2,              /**< side of the second (implied) inequality */
   SCIP_Bool             uselhs1,            /**< is the first inequality >=? */
   SCIP_Bool             uselhs2,            /**< is the second inequality >=? */
   SCIP_HASHMAP*         varmap,             /**< variable map */
   SCIP_Bool             f                   /**< the first relation is an implication x == f */
)
{
   SCIP_Real sign1;
   SCIP_Real sign2;
   SCIP_Real mult;
   SCIP_Real lincoefs[3];
   SCIP_VAR* w;
   SCIP_VAR* x;
   SCIP_VAR* y;
   SCIP_Bool overest; /* does linexpr overestimate the product? */
   SCIP_Real cst;

   /* x must be binary */
   assert(SCIPvarGetType(vars[0]) == SCIP_VARTYPE_BINARY);

   SCIPdebugMsg(scip, "Extracting product from two relations:\n");
   SCIPdebugMsg(scip, "Relation 1: %s == %d => %g%s + %g%s %s %g\n", SCIPvarGetName(vars[0]), f, coefs1[1],
      SCIPvarGetName(vars[1]), coefs1[2], SCIPvarGetName(vars[2]), uselhs1 ? ">=" : "<=", side1);
   SCIPdebugMsg(scip, "Relation 2: %s == %d => %g%s + %g%s %s %g\n", SCIPvarGetName(vars[0]), !f, coefs2[1],
      SCIPvarGetName(vars[1]), coefs2[2], SCIPvarGetName(vars[2]), uselhs2 ? ">=" : "<=", side2);

   assert( coefs1[0] != 0.0 ); /* the first relation is always conditional */

   x = vars[0];
   w = vars[1];
   y = vars[2];

   /* cannot use a global bound on x to detect a product */
   if( (coefs1[1] == 0.0 && coefs1[2] == 0.0) ||
       (coefs2[1] == 0.0 && coefs2[2] == 0.0) )
      return SCIP_OKAY;

   SCIPdebugMsg(scip, "binary var = %s, its coefs: %g\n", SCIPvarGetName(vars[0]), coefs1[0]*coefs2[0]);

   /* we flip the rows so that coefs of w are positive */
   sign1 = coefs1[1] >= 0 ? 1.0 : -1.0;
   sign2 = coefs2[1] >= 0 ? 1.0 : -1.0;

   /* flip the sides if needed */
   if( sign1 < 0 )
   {
      side1 *= -1.0;
      uselhs1 = !uselhs1;
   }

   if( sign2 < 0 )
   {
      side2 *= -1.0;
      uselhs2 = !uselhs2;
   }
   SCIPdebugMsg(scip, "\nsigns: %g, %g", sign1, sign2);

   if( uselhs1 != uselhs2 )
      return SCIP_OKAY;

   /* from here on, we consider only the flipped (by multiplying by signi) rows */

   /* at least one w coefficient must be nonzero */
   assert( coefs1[1] != 0.0 || coefs2[1] != 0.0 );

   /* cannot use a global bound on y to detect a non-redundant product relation */
   if( coefs2[0] == 0.0 && coefs2[1] == 0.0 ) /* only check the 2nd relation because the 1st at least has x */
   {
      SCIPdebugMsg(scip, "Ignoring a global bound on y\n");
      return SCIP_OKAY;
   }

   /* when a1c2 = a2c1, the linear relations do not imply a product relation */
   if( SCIPisZero(scip, coefs2[1]*sign2*coefs1[2]*sign1 - coefs2[2]*sign2*coefs1[1]*sign1) )
   {
      SCIPdebugMsg(scip, "Ignoring a pair of linear relations because a1c2 = a2c1\n");
      return SCIP_OKAY;
   }

   /* all conditions satisfied, we can extract the product */
   /* given two rows of the form:
    * a1w + b1x + c1y <= d1, a2w + b2x + c2y <= d2 (or same with >=),
    * where b1*b2 <= 0 and the first inequality is tighter when x = f and the second when x = !f,
    * and a1, a2 > 0, the product relation can be written as:
    * xy >=/<= (1/(a2c1 - c2a1))*(a1a2w + (a1(b2 - d2) + a2d1)x + a2c1y - a2d1) (if f == 0) or
    * xy >=/<= (1/(a1c2 - c1a2))*(a1a2w + (a2(b1 - d1) + a1d2)x + a1c2y - a1d2) (if f == 1)
    * (the inequality sign depends on the sign of (a1c2 - c1a2) and the sign in the linear inequalities) */
   /* TODO can swap coefs for one case and get rid of the else here */
   if( !f )
   {
      mult = 1/(coefs2[1]*sign2*coefs1[2]*sign1 - coefs2[2]*sign2*coefs1[1]*sign1);

      /* we make sure above that these have the same sign, but one of them might be zero, so we check both here */
      overest = mult < 0.0;
      if( uselhs1 ) /* only check uselhs1 because uselhs2 is equal to it */
         overest = !overest;

      SCIPdebugMsg(scip, "w coef is %s\n", overest ? "negative" : "positive");

      SCIPdebugMsg(scip, "!f, found suitable implied rels (w,x,y): %g%s + %g%s + %g%s >= %g\n", sign1*coefs1[1],
         SCIPvarGetName(w), sign1*coefs1[0], SCIPvarGetName(x), sign1*coefs1[2], SCIPvarGetName(y), side1);

      SCIPdebugMsg(scip, "\nand %g%s + %g%s + %g%s >= %g\n", sign2*coefs2[1], SCIPvarGetName(w), sign2*coefs2[0],
         SCIPvarGetName(x), sign2*coefs2[2], SCIPvarGetName(y), side2);

      lincoefs[0] = sign2*coefs2[1]*sign1*coefs1[1]*mult;
      lincoefs[1] = (-side2*sign1*coefs1[1] + side1*sign2*coefs2[1])*mult;
      lincoefs[2] = sign2*coefs2[1]*sign1*coefs1[2]*mult;

      SCIPdebugMsg(scip, "product: %s%s %s %g%s + %g%s + %g%s + %g\n", SCIPvarGetName(x), SCIPvarGetName(y), overest ? "<=" : ">=",
         lincoefs[0], SCIPvarGetName(w), lincoefs[1], SCIPvarGetName(x), lincoefs[2], SCIPvarGetName(y), -coefs2[1]*side1*mult);

      cst = -sign2*coefs2[1]*side1*mult;
   }
   else /* f == TRUE */
   {
      mult = 1/(coefs1[1]*sign1*coefs2[2]*sign2 - coefs1[2]*sign1*coefs2[1]*sign2);

      /* TODO we make sure above that these have the same sign, but one of them might be zero, so we check both here */
      overest = mult < 0.0;
      if( uselhs1 ) /* only check uselhs1 because uselhs2 is equal to it */
         overest = !overest;

      SCIPdebugMsg(scip, "w coef is %s\n", overest ? "negative" : "positive");

      SCIPdebugMsg(scip, "f, found suitable implied rels (w,x,y): %g%s + %g%s + %g%s <= %g\n", sign1*coefs1[1],
         SCIPvarGetName(w), sign1*coefs1[0], SCIPvarGetName(x), sign1*coefs1[2], SCIPvarGetName(y), side1);

      SCIPdebugMsg(scip, "\nand %g%s + %g%s + %g%s <= %g\n", sign2*coefs2[1], SCIPvarGetName(w),
         sign2*coefs2[0], SCIPvarGetName(x), sign2*coefs2[2], SCIPvarGetName(y), side2);

      lincoefs[0] = sign1*coefs1[1]*sign2*coefs2[1]*mult;
      lincoefs[1] = (-side1*sign2*coefs2[1] + side2*sign1*coefs1[1])*mult;
      lincoefs[2] = sign1*coefs1[1]*sign2*coefs2[2]*mult;

      SCIPdebugMsg(scip, "product: %s%s %s %g%s + %g%s + %g%s + %g\n", SCIPvarGetName(x), SCIPvarGetName(y), overest ? "<=" : ">=",
         lincoefs[0], SCIPvarGetName(w), lincoefs[1], SCIPvarGetName(x), lincoefs[2], SCIPvarGetName(y), -coefs1[1]*side2*mult);

      cst = -sign1*coefs1[1]*side2*mult;
   }

   SCIP_CALL( addProductVars(scip, sepadata, x, y, varmap, 1) );

   SCIP_CALL( SCIPinsertBilinearTermImplicit(scip, sepadata->conshdlr, x, y, w, lincoefs[0], lincoefs[1], lincoefs[2],
         cst, overest) );

   return SCIP_OKAY;
}

/** extract products from a relation given by coefs1, vars, lhs1 and rhs1 and
 *  implied bounds of the form vars[varpos1] == !f => vars[varpos2] >=/<= bound
 */
static
SCIP_RETCODE detectProductsImplbnd(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_Real*            coefs1,             /**< coefficients of the first linear relation */
   SCIP_VAR**            vars,               /**< variables of the first relation in the order x, w, y */
   SCIP_Real             side1,              /**< side of the first relation */
   SCIP_Bool             uselhs1,            /**< is the left (TRUE) or right (FALSE) hand side given for the first relation? */
   int                   varpos1,            /**< position of the indicator variable in the vars array */
   int                   varpos2,            /**< position of the variable that is bounded */
   SCIP_HASHMAP*         varmap,             /**< variable map */
   SCIP_Bool             f                   /**< the value of x that activates the first relation */
   )
{
   SCIP_Real coefs2[3];
   SCIP_Bool foundlb;
   SCIP_Bool foundub;
   SCIP_Real impllb;
   SCIP_Real implub;
   SCIP_VAR* var1;
   SCIP_VAR* var2;
   SCIP_Bool xval;
   SCIP_Bool globalside;

   assert( varpos1 != varpos2 );

   var1 = vars[varpos1];
   var2 = vars[varpos2];

   assert(SCIPvarGetType(var1) == SCIP_VARTYPE_BINARY && SCIPvarGetType(var2) != SCIP_VARTYPE_BINARY);

   xval = !f;

   /* if it is an unconditional relation (i.e. in w and y) and xval = 1, then global bound is used as side
    * (since the big-M inequality is then written as var2 - (implbnd - globbnd)var1 <=/>= globbnd) */
   globalside = varpos1 != 0 && xval;

   if( varpos1 != 2 && varpos2 != 2 )
      coefs2[2] = 0.0;
   else if( varpos1 != 1 && varpos2 != 1 )
      coefs2[1] = 0.0;
   else
      coefs2[0] = 0.0;

   coefs2[varpos2] = 1.0;

   /* get implications var1 == xval  =>  var2 <= implub (or var2 >= implub) */
   SCIPvarGetImplicVarBounds(var1, xval, var2, &impllb, &implub, &foundlb, &foundub);

   if( foundlb )
   {
      coefs2[varpos1] = SCIPvarGetLbGlobal(var2) - impllb;
      if( !xval )
         coefs2[varpos1] *= -1.0;
      SCIP_CALL( extractProducts(scip, sepadata, vars, coefs1, coefs2, side1,
         globalside ? SCIPvarGetLbGlobal(var2) : impllb, uselhs1, TRUE, varmap, f) );
   }

   if( foundub )
   {
      coefs2[varpos1] = SCIPvarGetUbGlobal(var2) - implub;
      if( !xval )
         coefs2[varpos1] *= -1.0;
      SCIP_CALL( extractProducts(scip, sepadata, vars, coefs1, coefs2, side1,
         globalside ? SCIPvarGetUbGlobal(var2) : implub, uselhs1, FALSE, varmap, f) );
   }

   return SCIP_OKAY;
}

/** extract products from a relation given by coefs1, vars, lhs1 and rhs1 and
 *  cliques containing vars[varpos1] and vars[varpos2]
 */
static
SCIP_RETCODE detectProductsClique(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_Real*            coefs1,             /**< coefficients of the first linear relation */
   SCIP_VAR**            vars,               /**< variables of the first relation in the order x, w, y */
   SCIP_Real             side1,              /**< side of the first relation */
   SCIP_Bool             uselhs1,            /**< is the left (TRUE) or right (FALSE) hand side given for the first relation? */
   int                   varpos1,            /**< position of the first (binary) variable in the vars array */
   int                   varpos2,            /**< position of the second (binary) variable in the vars array */
   SCIP_HASHMAP*         varmap,             /**< variable map */
   SCIP_Bool             f                   /**< the value of x that activates the first relation */
)
{
   SCIP_Real coefs2[3];
   SCIP_VAR* var1;
   SCIP_VAR* var2;
   SCIP_Bool xval;
   SCIP_Real side;

   var1 = vars[varpos1];
   var2 = vars[varpos2];
   xval = !f;

   assert(SCIPvarGetType(var1) == SCIP_VARTYPE_BINARY && SCIPvarGetType(var2) == SCIP_VARTYPE_BINARY);

   if( varpos1 != 2 && varpos2 != 2 )
      coefs2[2] = 0.0;
   else if( varpos1 != 1 && varpos2 != 1 )
      coefs2[1] = 0.0;
   else
      coefs2[0] = 0.0;

   /* if both vals are TRUE, the relation is var1 + var2 <= 1
    * for a FALSE val: var is changed to (1-var) => reverse the coef, substract 1 from side */
   if( SCIPvarsHaveCommonClique(var1, xval, var2, TRUE, TRUE) )
   {
      SCIPdebugMsg(scip, "vars %s%s and %s are in a clique\n", xval ? "" : "!", SCIPvarGetName(var1), SCIPvarGetName(var2));
      coefs2[varpos1] = xval ? 1.0 : -1.0;
      coefs2[varpos2] = 1.0;

      if( varpos1 != 0 && xval ) /* relation is unconditional and clique has var1 */
         side = 1.0;
      else
         side = 0.0;

      SCIP_CALL( extractProducts(scip, sepadata, vars, coefs1, coefs2, side1, side, uselhs1, FALSE, varmap, f) );
   }
   if( SCIPvarsHaveCommonClique(var1, xval, var2, FALSE, TRUE) )
   {
      SCIPdebugMsg(scip, "vars %s%s and !%s are in a clique\n", xval ? "" : "!", SCIPvarGetName(var1), SCIPvarGetName(var2));
      coefs2[varpos1] = xval ? 1.0 : -1.0;
      coefs2[varpos2] = -1.0;

      if( varpos1 != 0 && xval ) /* relation is unconditional and clique has var1 */
         side = 0.0;
      else
         side = -1.0;

      SCIP_CALL( extractProducts(scip, sepadata, vars, coefs1, coefs2, side1, side, uselhs1, FALSE, varmap, f) );
   }

   return SCIP_OKAY;
}


/** extract products from a relation given by coefs1, vars, lhs1 and rhs1 and unconditional relations
 * (inequalities with 2 nonzeroes) containing vars[varpos1] and vars[varpos2]
 */
static
SCIP_RETCODE detectProductsUnconditional(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_ROW**            rows,               /**< problem rows */
   int*                  row_list,           /**< linked list of rows corresponding to 2 or 3 var sets */
   SCIP_HASHTABLE*       hashtable,          /**< hashtable storing unconditional relations */
   SCIP_Real*            coefs1,             /**< coefficients of the first linear relation */
   SCIP_VAR**            vars,               /**< variables of the first relation in the order x, w, y */
   SCIP_Real             side1,              /**< side of the first relation */
   SCIP_Bool             uselhs1,            /**< is the left (TRUE) or right (FALSE) hand side given for the first relation? */
   int                   varpos1,            /**< position of the first unconditional variable in the vars array */
   int                   varpos2,            /**< position of the second unconditional variable in the vars array */
   SCIP_HASHMAP*         varmap,             /**< variable map */
   SCIP_Bool             f,                  /**< the value of x that activates the first relation */
   SCIP_Bool             fixedside           /**< indicates if there is no need to substract the big-M from side1 */
   )
{
   HASHDATA hashdata;
   HASHDATA* foundhashdata;
   SCIP_ROW* row2;
   int r2;
   int pos1;
   int pos2;
   SCIP_Real coefs2[3];
   SCIP_VAR* var1;
   SCIP_VAR* var2;

   assert(varpos1 != 0); /* always unconditional */

   if( varpos1 != 2 && varpos2 != 2 )
      coefs2[2] = 0.0;
   else if( varpos1 != 1 && varpos2 != 1 )
      coefs2[1] = 0.0;
   else
      coefs2[0] = 0.0;

   var1 = vars[varpos1];
   var2 = vars[varpos2];

   hashdata.nvars = 2;
   hashdata.firstrow = -1;
   SCIP_CALL( SCIPallocBufferArray(scip, &(hashdata.vars), 2) );
   if( SCIPvarGetIndex(var1) < SCIPvarGetIndex(var2) )
   {
      pos1 = 0;
      pos2 = 1;
   }
   else
   {
      pos1 = 1;
      pos2 = 0;
   }

   hashdata.vars[pos1] = var1;
   hashdata.vars[pos2] = var2;

   foundhashdata = (HASHDATA*)SCIPhashtableRetrieve(hashtable, &hashdata);

   if( foundhashdata != NULL )
   {
      /* if the var pair exists, use all corresponding rows */
      r2 = foundhashdata->firstrow;

      while( r2 != -1 )
      {
         row2 = rows[r2];
         assert(SCIProwGetNNonz(row2) == 2);
         assert(var1 == SCIPcolGetVar(SCIProwGetCols(row2)[pos1]));
         assert(var2 == SCIPcolGetVar(SCIProwGetCols(row2)[pos2]));

         coefs2[varpos1] = SCIProwGetVals(row2)[pos1];
         coefs2[varpos2] = SCIProwGetVals(row2)[pos2];

         SCIPdebugMsg(scip, "Unconditional:\n");
         if( !SCIPisInfinity(scip,-SCIProwGetLhs(row2)) )
            SCIP_CALL( extractProducts(scip, sepadata, vars, coefs1, coefs2, (f && !fixedside) ? side1-coefs1[0] : side1, SCIProwGetLhs(row2) - SCIProwGetConstant(row2), uselhs1, TRUE, varmap, f) );
         if( !SCIPisInfinity(scip, SCIProwGetRhs(row2)) )
            SCIP_CALL( extractProducts(scip, sepadata, vars, coefs1, coefs2, (f && !fixedside) ? side1-coefs1[0] : side1, SCIProwGetRhs(row2) - SCIProwGetConstant(row2), uselhs1, FALSE, varmap, f) );

         r2 = row_list[r2];
      }
   }
   SCIPfreeBufferArray(scip, &(hashdata.vars));

   return SCIP_OKAY;
}

/**
 * stores implied relations (x == f => ay + bw <= c, f can be 0 or 1) and 2-variable relations in hashtables
 */
static
SCIP_RETCODE createRelationTables(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            prob_rows,          /**< linear rows of the problem */
   int                   nrows,              /**< number of rows */
   SCIP_HASHTABLE*       hashtable2,         /**< hashtable to store 2-variable relations */
   SCIP_HASHTABLE*       hashtable3,         /**< hashtable to store implied relations */
   SCIP_VAR**            vars_in_2rels,      /**< array of variables that appear in 2-variable relations */
   SCIP_VAR***           related_vars,       /**< for each var in vars_in_2rels, array of related vars */
   int*                  nrelated_vars,      /**< for each var in vars_in_2rels, number of related vars */
   int*                  nvars_in_2rels,     /**< number of variables that appear in 2-variable relations */
   int*                  row_list            /**< linked lists of rows for each 2 or 3 variable set */
   )
{
   int r;
   int v1;
   int v2;
   int i;
   int varpos1;
   int varpos2;
   SCIP_COL** cols;
   HASHDATA hashdata;
   HASHDATA* foundhashdata;
   SCIP_Bool found;

   *nvars_in_2rels = 0;

   /* initialise row_list */
   for( r = 0; r < nrows; ++r )
   {
      row_list[r] = -1;
   }

   /* look for implied relations and unconditional relations with 2 variables */
   for( r = 0; r < nrows; ++r ) /* go through rows */
   {
      assert(prob_rows[r] != NULL);

      cols = SCIProwGetCols(prob_rows[r]);
      assert(cols != NULL);
      SCIPdebugMsg(scip, "row %s:\n", SCIProwGetName(prob_rows[r]));
#ifdef SCIP_DEBUG
      for( v1 = 0; v1 < SCIProwGetNNonz(prob_rows[r]); ++v1 )
         SCIPdebugMsg(scip,"%s(%d) \n", SCIPvarGetName(SCIPcolGetVar(cols[v1])), SCIPcolGetIndex(cols[v1]));
#endif

      /* look for unconditional relations with 2 variables */
      if( SCIProwGetNNonz(prob_rows[r]) == 2 ) /* this can be an unconditional relation */
      {
         /* if at least one of the variables is binary, this is either an implied bound
          * or a clique; these are covered separately */
         /* TODO is that so? */
         if( SCIPvarGetType(SCIPcolGetVar(cols[0])) == SCIP_VARTYPE_BINARY
          || SCIPvarGetType(SCIPcolGetVar(cols[1])) == SCIP_VARTYPE_BINARY)
         {
            SCIPdebugMsg(scip, "ignoring relation %s because a var is binary\n", SCIProwGetName(prob_rows[r]));
            continue;
         }

         /* fill in hashdata */
         hashdata.nvars = 2;
         hashdata.firstrow = -1;
         SCIP_CALL( SCIPallocBufferArray(scip, &(hashdata.vars), 2) );
         for( v1 = 0; v1 < 2; ++v1 )
         {
            hashdata.vars[v1] = SCIPcolGetVar(cols[v1]);
         }

         /* get the element corresponsing to the two variables */
         foundhashdata = (HASHDATA*)SCIPhashtableRetrieve(hashtable2, &hashdata);

         if( foundhashdata != NULL )
         {
            /* if element exists, update it by adding the row */
            row_list[r] = foundhashdata->firstrow;
            foundhashdata->firstrow = r;

            ++foundhashdata->nrows;
            SCIPfreeBufferArray(scip, &(hashdata.vars));
         }
         else
         {
            /* create an element for the combination of two variables */
            SCIP_CALL( SCIPallocBuffer(scip, &foundhashdata) );

            foundhashdata->nvars = 2;
            foundhashdata->nrows = 1;
            foundhashdata->vars = hashdata.vars;

            foundhashdata->firstrow = r;

            SCIP_CALL( SCIPhashtableInsert(hashtable2, (void*)foundhashdata) );

            /* update the variable arrays */
            for( v1 = 0; v1 < 2; ++v1 )
            {
               v2 = 1 - v1;

               found = SCIPsortedvecFindPtr((void**)vars_in_2rels, SCIPvarComp, hashdata.vars[v1], *nvars_in_2rels, &varpos1);

               if( found )
               {
                  assert(vars_in_2rels[varpos1] == hashdata.vars[v1]);

                  /* add second var to corresponding array */
                  found = SCIPsortedvecFindPtr((void**)related_vars[varpos1], SCIPvarComp, hashdata.vars[v2], nrelated_vars[varpos1], &varpos2);
                  if( !found )
                  {
                     /* insert var at the correct position */
                     for( i = nrelated_vars[varpos1]; i > varpos2; --i )
                        related_vars[varpos1][i] = related_vars[varpos1][i-1];
                     related_vars[varpos1][varpos2] = hashdata.vars[v2];
                     nrelated_vars[varpos1]++;
                  }
               }
               else
               {  /* var has not been added yet, add it here */

                  /* insert expression at the correct position */
                  for( i = *nvars_in_2rels; i > varpos1; --i )
                  {
                     vars_in_2rels[i] = vars_in_2rels[i-1];
                     related_vars[i] = related_vars[i-1];
                     nrelated_vars[i] = nrelated_vars[i-1];
                  }
                  vars_in_2rels[varpos1] = hashdata.vars[v1];

                  SCIP_CALL( SCIPallocBufferArray(scip, &related_vars[varpos1], SCIPgetNVars(scip)) );
                  related_vars[varpos1][0] = hashdata.vars[v2];
                  nrelated_vars[varpos1] = 1;
                  ++(*nvars_in_2rels);
               }
            }
         }
      }

      /* look for implied relations */
      if( SCIProwGetNNonz(prob_rows[r]) == 3 )
      {
         /* an implied relation contains at least one binary variable */
         if( SCIPvarGetType(SCIPcolGetVar(cols[0])) != SCIP_VARTYPE_BINARY
          && SCIPvarGetType(SCIPcolGetVar(cols[1])) != SCIP_VARTYPE_BINARY
          && SCIPvarGetType(SCIPcolGetVar(cols[2])) != SCIP_VARTYPE_BINARY )
            continue;

         /* fill in hashdata */
         hashdata.nvars = 3;
         hashdata.firstrow = -1;
         SCIP_CALL( SCIPallocBufferArray(scip, &(hashdata.vars), 3) );
         for( v1 = 0; v1 < 3; ++v1 )
         {
            hashdata.vars[v1] = SCIPcolGetVar(cols[v1]);
         }

         /* get the element corresponsing to the three variables */
         foundhashdata = (HASHDATA*)SCIPhashtableRetrieve(hashtable3, &hashdata);

         if( foundhashdata != NULL )
         {
            /* if element exists, update it by adding the row */
            row_list[r] = foundhashdata->firstrow;
            foundhashdata->firstrow = r;

            ++foundhashdata->nrows;
            SCIPfreeBufferArray(scip, &(hashdata.vars));
         }
         else
         {
            /* create an element for the combination of three variables */
            SCIP_CALL( SCIPallocBuffer(scip, &foundhashdata) );

            foundhashdata->nvars = 3;
            foundhashdata->nrows = 1;
            foundhashdata->vars = hashdata.vars;

            foundhashdata->firstrow = r;

            SCIP_CALL( SCIPhashtableInsert(hashtable3, (void*)foundhashdata) );
         }
      }
   }

   return SCIP_OKAY;
}

/* detect bilinear products encoded in linear constraints */
static
SCIP_RETCODE detectHiddenProducts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separation data */
   SCIP_HASHMAP*         varmap              /**< variable map */
   )
{
   int r1; /* first relation index */
   int r2; /* second relation index */
   int i; /* outer loop counter */
   int permwy; /* index for permuting w and y */
   int nrows;
   SCIP_ROW** prob_rows;
   SCIP_HASHTABLE* hashtable3;
   SCIP_HASHTABLE* hashtable2;
   HASHDATA* foundhashdata;
   SCIP_VAR* vars_xwy[3];
   SCIP_Real coefs1[3];
   SCIP_Real coefs2[3];
   SCIP_ROW* row1;
   SCIP_ROW* row2;
   int xpos;
   int ypos;
   int wpos;
   int f; /* value of the binary variable */
   int nvars_in_2rels;
   int varpos;
   SCIP_VAR** vars_in_2rels;
   SCIP_VAR*** related_vars;
   int* nrelated_vars;
   SCIP_Bool found;
   SCIP_Bool xfixing;
   SCIP_Bool uselhs1;
   SCIP_Bool uselhs2;
   SCIP_Real side1;
   SCIP_Real side2;
   int* row_list;

   /* get the (initial) rows */
   SCIP_CALL( getInitialRows(scip, &prob_rows, &nrows) );

   /* create tables of implied and unconditional relations */
   SCIP_CALL( SCIPhashtableCreate(&hashtable3, SCIPblkmem(scip), nrows, SCIPhashGetKeyStandard,
      hashdataKeyEqConss, hashdataKeyValConss, (void*) scip) );
   SCIP_CALL( SCIPhashtableCreate(&hashtable2, SCIPblkmem(scip), nrows, SCIPhashGetKeyStandard,
      hashdataKeyEqConss, hashdataKeyValConss, (void*) scip) );
   SCIP_CALL( SCIPallocBufferArray(scip, &row_list, nrows) );

   /* allocate the array of variables that appear in 2-var relations */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars_in_2rels, SCIPgetNVars(scip)) );
   /* allocate the array of arrays of variables that appear in 2-var relations with each variable */
   SCIP_CALL( SCIPallocBufferArray(scip, &related_vars, SCIPgetNVars(scip)) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nrelated_vars, SCIPgetNVars(scip)) );

   /* add relations to hashtables */
   SCIP_CALL( createRelationTables(scip, prob_rows, nrows, hashtable2, hashtable3, vars_in_2rels, related_vars,
      nrelated_vars, &nvars_in_2rels, row_list) );

#ifdef SCIP_DEBUG
   SCIPinfoMessage(scip, NULL, "\nrelated vars:");
   for( i = 0; i < nvars_in_2rels; ++i )
   {
      int j;

      SCIPinfoMessage(scip, NULL, "\nfor var %s: ", SCIPvarGetName(vars_in_2rels[i]));
      for( j = 0; j < nrelated_vars[i]; ++j )
         SCIPinfoMessage(scip, NULL, "%s; ", SCIPvarGetName(related_vars[i][j]));
   }
   SCIPinfoMessage(scip, NULL, "\n");
   SCIPinfoMessage(scip, NULL, "\nImplied relations table:\n");
#endif

   SCIP_CALL( SCIPcreateClock(scip, &sepadata->cliquetime) );
   SCIP_CALL( SCIPresetClock(scip, sepadata->cliquetime) );

   /* start actually looking for products */
   /* go through all sets of three variables */
   for( i = 0; i < SCIPhashtableGetNEntries(hashtable3); ++i )
   {
      foundhashdata = (HASHDATA*)SCIPhashtableGetEntry(hashtable3, i);
      if( foundhashdata == NULL )
         continue;

      SCIPdebugMsg(scip, "(%s, %s, %s): ", SCIPvarGetName(foundhashdata->vars[0]),
         SCIPvarGetName(foundhashdata->vars[1]), SCIPvarGetName(foundhashdata->vars[2]));

      /* an implied relation has the form: x = f => w <= ay + b (f is 0 or 1)
       * given an arbitrary linear relation, any binary var can be x
       * we try them all here because this can produce different products */
      for( xpos = 0; xpos < 3; ++xpos )
      {
         vars_xwy[0] = foundhashdata->vars[xpos];

         /* x must be binary */
         if( SCIPvarGetType(vars_xwy[0]) != SCIP_VARTYPE_BINARY )
            continue;

         /* the first row might be an implication from f == 0 or f == 1: try both */
         for( f = 0; f <= 1; ++f )
         {
            xfixing = f == 1;

            /* go through implied relations for the corresponding three variables */
            r1 = foundhashdata->firstrow;
            while( r1 != -1 )
            {
               row1 = prob_rows[r1];

               assert(SCIProwGetNNonz(row1) == 3);
               assert(vars_xwy[0] == SCIPcolGetVar(SCIProwGetCols(row1)[xpos]));

               coefs1[0] = SCIProwGetVals(row1)[xpos];

               if( (!xfixing && coefs1[0] > 0) ||  (xfixing && coefs1[0] < 0) )
               {
                  uselhs1 = TRUE;
                  side1 = SCIProwGetLhs(row1);
               }
               else
               {
                  uselhs1 = FALSE;
                  side1 = SCIProwGetRhs(row1);
               }

               if( SCIPisInfinity(scip, REALABS(side1)) )
               {
                  r1 = row_list[r1];
                  continue;
               }

               side1 -= SCIProwGetConstant(row1);

               /* permute w and y */
               for( permwy = 1; permwy <= 2; ++permwy )
               {
                  wpos = (xpos + permwy) % 3;
                  ypos = (xpos - permwy + 3) % 3;
                  vars_xwy[1] = foundhashdata->vars[wpos];
                  vars_xwy[2] = foundhashdata->vars[ypos];

                  assert(vars_xwy[1] == SCIPcolGetVar(SCIProwGetCols(row1)[wpos]));
                  assert(vars_xwy[2] == SCIPcolGetVar(SCIProwGetCols(row1)[ypos]));

                  coefs1[1] = SCIProwGetVals(row1)[wpos];
                  coefs1[2] = SCIProwGetVals(row1)[ypos];

                  /* look for the second relation */
                  /* the second relation should be active when x == !f */

                  /* go through the remaining rows for these three variables */
                  r2 = row_list[r1];

                  while( r2 != -1 )
                  {
                     row2 = prob_rows[r2];

                     assert(SCIProwGetNNonz(row2) == 3);
                     assert(vars_xwy[0] == SCIPcolGetVar(SCIProwGetCols(row2)[xpos]));
                     assert(vars_xwy[1] == SCIPcolGetVar(SCIProwGetCols(row2)[wpos]));
                     assert(vars_xwy[2] == SCIPcolGetVar(SCIProwGetCols(row2)[ypos]));

                     coefs2[0] = SCIProwGetVals(row2)[xpos];
                     coefs2[1] = SCIProwGetVals(row2)[wpos];
                     coefs2[2] = SCIProwGetVals(row2)[ypos];

                     if( (!xfixing && coefs2[0] > 0) ||  (xfixing && coefs2[0] < 0) )
                     {
                        uselhs2 = FALSE;
                        side2 = SCIProwGetRhs(row2);
                     }
                     else
                     {
                        uselhs2 = TRUE;
                        side2 = SCIProwGetLhs(row2);
                     }

                     if( SCIPisInfinity(scip, REALABS(side2)) )
                     {
                        r2 = row_list[r2];
                        continue;
                     }

                     side2 -= SCIProwGetConstant(row2);

                     SCIPdebugMsg(scip, "Two implied relations:\n");
                     SCIP_CALL( extractProducts(scip, sepadata, vars_xwy, coefs1, coefs2, xfixing ? side1-coefs1[0] : side1,
                        !xfixing ? side2-coefs2[0] : side2, uselhs1, uselhs2, varmap, xfixing) );

                     r2 = row_list[r2];
                  }

                  /* use global bounds on w */
                  coefs2[0] = 0.0;
                  coefs2[1] = 1.0;
                  coefs2[2] = 0.0;
                  SCIPdebugMsg(scip, "w global bounds:\n");
                  if( !SCIPisInfinity(scip, -SCIPvarGetLbGlobal(vars_xwy[1])) )
                     SCIP_CALL( extractProducts(scip, sepadata, vars_xwy, coefs1, coefs2, xfixing ? side1-coefs1[0] : side1,
                        SCIPvarGetLbGlobal(vars_xwy[1]), uselhs1, TRUE, varmap, xfixing) );

                  if( !SCIPisInfinity(scip, SCIPvarGetUbGlobal(vars_xwy[1])) )
                     SCIP_CALL( extractProducts(scip, sepadata, vars_xwy, coefs1, coefs2, xfixing ? side1-coefs1[0] : side1,
                        SCIPvarGetUbGlobal(vars_xwy[1]), uselhs1, FALSE, varmap, xfixing) );

                  /* do implied bounds and cliques with w */
                  if( SCIPvarGetType(vars_xwy[1]) != SCIP_VARTYPE_BINARY )
                  { /* w is non-binary - look for implied bounds x = !f => w >=/<= bound */
                     SCIPdebugMsg(scip, "Implied relation + implied bounds on w:\n");
                     SCIP_CALL( detectProductsImplbnd(scip, sepadata, coefs1, vars_xwy, xfixing ? side1-coefs1[0] : side1, uselhs1, 0, 1, varmap, xfixing) );
                  }
                  else
                  { /* w is binary - look for cliques containing x and w */
                     SCIPdebugMsg(scip, "Implied relation + cliques with x and w:\n");
                     SCIP_CALL( SCIPstartClock(scip, sepadata->cliquetime) );
                     SCIP_CALL( detectProductsClique(scip, sepadata, coefs1, vars_xwy, xfixing ? side1-coefs1[0] : side1, uselhs1, 0, 1, varmap, xfixing) );
                     SCIP_CALL( SCIPstopClock(scip, sepadata->cliquetime) );
                  }

                  /* use unconditional relations (i.e. relations of w and y) */

                  /* implied bound w == 0/1 => y >=/<= bound */
                  if( SCIPvarGetType(vars_xwy[1]) == SCIP_VARTYPE_BINARY && SCIPvarGetType(vars_xwy[2]) != SCIP_VARTYPE_BINARY )
                  {
                     SCIPdebugMsg(scip, "Implied relation + implied bounds with w and y:\n");
                     SCIP_CALL( detectProductsImplbnd(scip, sepadata, coefs1, vars_xwy, side1, uselhs1, 1, 2, varmap, FALSE) );
                     SCIP_CALL( detectProductsImplbnd(scip, sepadata, coefs1, vars_xwy, side1-coefs1[0], uselhs1, 1, 2, varmap, TRUE) );
                  }

                  /* implied bound y == 0/1 => w >=/<= bound */
                  if( SCIPvarGetType(vars_xwy[2]) == SCIP_VARTYPE_BINARY && SCIPvarGetType(vars_xwy[1]) != SCIP_VARTYPE_BINARY )
                  {
                     SCIPdebugMsg(scip, "Implied relation + implied bounds with y and w:\n");
                     SCIP_CALL( detectProductsImplbnd(scip, sepadata, coefs1, vars_xwy, side1, uselhs1, 2, 1, varmap, FALSE) );
                     SCIP_CALL( detectProductsImplbnd(scip, sepadata, coefs1, vars_xwy, side1-coefs1[0], uselhs1, 2, 1, varmap, TRUE) );
                  }

                  /* cliques containing w and y */
                  if( SCIPvarGetType(vars_xwy[1]) == SCIP_VARTYPE_BINARY && SCIPvarGetType(vars_xwy[2]) == SCIP_VARTYPE_BINARY )
                  {
                     SCIPdebugMsg(scip, "Implied relation + cliques with w and y:\n"); /* TODO this can be more efficient with x in outer loop */
                     SCIP_CALL( SCIPstartClock(scip, sepadata->cliquetime) );
                     SCIP_CALL( detectProductsClique(scip, sepadata, coefs1, vars_xwy, side1, uselhs1, 1, 2, varmap, FALSE) );
                     SCIP_CALL( detectProductsClique(scip, sepadata, coefs1, vars_xwy, side1-coefs1[0], uselhs1, 1, 2, varmap, TRUE) );
                     SCIP_CALL( SCIPstopClock(scip, sepadata->cliquetime) );
                  }

                  /* inequalities containing w and y */
                  if( SCIPvarGetType(vars_xwy[1]) != SCIP_VARTYPE_BINARY && SCIPvarGetType(vars_xwy[2]) != SCIP_VARTYPE_BINARY )
                  {
                     SCIPdebugMsg(scip, "Implied relation + unconditional with w and y:\n");
                     SCIP_CALL( detectProductsUnconditional(scip, sepadata, prob_rows, row_list, hashtable2, coefs1, vars_xwy, side1, uselhs1,
                        1, 2, varmap, xfixing, FALSE) );
                  }
               }
               r1 = row_list[r1];
            }
         }
      }
      SCIPfreeBufferArray(scip, &(foundhashdata->vars));
      SCIPfreeBuffer(scip, &foundhashdata);
   }


   /* also loop through implied bounds to look for products */
   for( i = 0; i < SCIPgetNVars(scip); ++i )
   {
      vars_xwy[0] = SCIPgetVars(scip)[i];

      if( SCIPvarGetType(vars_xwy[0]) != SCIP_VARTYPE_BINARY )
         continue;

      coefs1[1] = 1.0;
      coefs1[2] = 0.0;

      for( f = 0; f <= 1; ++f )
      {
         xfixing = f == 1;

         /* go through implications of x */
         for( r1 = 0; r1 < SCIPvarGetNImpls(vars_xwy[0], xfixing); ++r1 )
         {
            /* w is the implic var */
            /* y could be anything, but must be in relation with w */
            vars_xwy[1] = SCIPvarGetImplVars(vars_xwy[0], xfixing)[r1];

            if( SCIPvarGetImplTypes(vars_xwy[0], xfixing)[r1] == SCIP_BOUNDTYPE_LOWER )
            {
               coefs1[0] = SCIPvarGetLbGlobal(vars_xwy[1]) - SCIPvarGetImplBounds(vars_xwy[0], xfixing)[r1];
               side1 = SCIPvarGetImplBounds(vars_xwy[0], xfixing)[r1];
               uselhs1 = TRUE;
            }
            else
            {
               coefs1[0] = SCIPvarGetUbGlobal(vars_xwy[1]) - SCIPvarGetImplBounds(vars_xwy[0], xfixing)[r1];
               side1 = SCIPvarGetImplBounds(vars_xwy[0], xfixing)[r1];
               uselhs1 = FALSE;
            }
            if( !xfixing )
               coefs1[0] = -coefs1[0];

            /* if the global bound is equal to the implied bound, there is nothing to do */
            if( coefs1[0] == 0.0 )
               continue;

            /* the second relation is in w and y */
            coefs2[0] = 0.0;

            assert(SCIPvarGetType(vars_xwy[1]) != SCIP_VARTYPE_BINARY);

            SCIPdebugMsg(scip, "Implic of x = %s + implied lb on w = %s:\n", SCIPvarGetName(vars_xwy[0]), SCIPvarGetName(vars_xwy[1]));

            /* use implied lower bounds on w: w >= b*y + d */
            for( r2 = 0; r2 < SCIPvarGetNVlbs(vars_xwy[1]); ++r2 )
            {
               vars_xwy[2] = SCIPvarGetVlbVars(vars_xwy[1])[r2];
               if( vars_xwy[2] == vars_xwy[0] )
                  continue;

               coefs2[1] = 1.0;
               coefs2[2] = -SCIPvarGetVlbCoefs(vars_xwy[1])[r2];

               SCIP_CALL( extractProducts(scip, sepadata, vars_xwy, coefs1, coefs2, side1,
                  SCIPvarGetVlbConstants(vars_xwy[1])[r2], uselhs1, TRUE, varmap, xfixing) );
            }

            SCIPdebugMsg(scip, "Implic of x = %s + implied ub on w = %s:\n", SCIPvarGetName(vars_xwy[0]), SCIPvarGetName(vars_xwy[1]));

            /* use implied upper bounds on w: w <= b*y + d */
            for( r2 = 0; r2 < SCIPvarGetNVubs(vars_xwy[1]); ++r2 )
            {
               vars_xwy[2] = SCIPvarGetVubVars(vars_xwy[1])[r2];
               if( vars_xwy[2] == vars_xwy[0] )
                  continue;

               coefs2[1] = 1.0;
               coefs2[2] = -SCIPvarGetVubCoefs(vars_xwy[1])[r2];

               SCIP_CALL( extractProducts(scip, sepadata, vars_xwy, coefs1, coefs2, side1,
                  SCIPvarGetVubConstants(vars_xwy[1])[r2], uselhs1, FALSE, varmap, xfixing) );
            }

            /* use unconditional relations containing w */
            found = SCIPsortedvecFindPtr((void**)vars_in_2rels, SCIPvarComp, vars_xwy[1], nvars_in_2rels, &varpos);
            if( !found )
               continue;

            for( r2 = 0; r2 < nrelated_vars[varpos]; ++r2 )
            {
               vars_xwy[2] = related_vars[varpos][r2];
               SCIPdebugMsg(scip, "Implied bound + unconditional with w and y:\n");
               SCIP_CALL( detectProductsUnconditional(scip, sepadata, prob_rows, row_list, hashtable2, coefs1, vars_xwy, side1,
                  uselhs1, 1, 2, varmap, xfixing, TRUE) );
            }
         }
      }
   }

   SCIPinfoMessage(scip, NULL, "\nTime spent on handling cliques: %10.2f", SCIPgetClockTime(scip, sepadata->cliquetime));
   SCIP_CALL( SCIPfreeClock(scip, &sepadata->cliquetime) );

   for( i = 0; i < nvars_in_2rels; ++i )
   {
      SCIPfreeBufferArray(scip, &related_vars[i]);
   }

   SCIPdebugMsg(scip, "Unconditional relations table:\n");
   for( i = 0; i < SCIPhashtableGetNEntries(hashtable2); ++i )
   {
      foundhashdata = (HASHDATA*)SCIPhashtableGetEntry(hashtable2, i);
      if( foundhashdata == NULL )
         continue;

      SCIPdebugMsg(scip, "(%s, %s): ", SCIPvarGetName(foundhashdata->vars[0]),
                   SCIPvarGetName(foundhashdata->vars[1]));

      SCIPfreeBufferArray(scip, &(foundhashdata->vars));
      SCIPfreeBuffer(scip, &foundhashdata);
   }

   SCIPfreeBufferArray(scip, &nrelated_vars);
   SCIPfreeBufferArray(scip, &related_vars);
   SCIPfreeBufferArray(scip, &vars_in_2rels);

   SCIPfreeBufferArray(scip, &row_list);

   SCIPhashtableFree(&hashtable2);
   SCIPhashtableFree(&hashtable3);

   SCIPfreeBufferArray(scip, &prob_rows);

   return SCIP_OKAY;
}

/** helper method to create separation data */
static
SCIP_RETCODE createSepaData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata            /**< separation data */
   )
{
   SCIP_HASHMAP* varmap;
   int i;
   int nvars;
   SCIP_CONSEXPR_BILINTERM* bilinterms;
   int nbilinterms;

   assert(sepadata != NULL);

   /* get total number of bilinear terms */
   nbilinterms = SCIPgetConsExprNBilinTerms(sepadata->conshdlr);

   /* skip if there are no bilinear terms and implicit product detection is off */
   if( nbilinterms == 0 && sepadata->detecthidden == FALSE )
   {
      return SCIP_OKAY;
   }

   nvars = SCIPgetNVars(scip);

   /* create variable map */
   SCIP_CALL( SCIPhashmapCreate(&varmap, SCIPblkmem(scip), SCIPgetNVars(scip)) );

   /* create the empty map for bilinear terms */
   SCIP_CALL( SCIPhashmapCreate(&sepadata->bilinvarsmap, SCIPblkmem(scip), nvars) );

   /* initialise some fields of sepadata */
   sepadata->varssorted = NULL;
   sepadata->bilinvardatas = NULL;
   sepadata->eqlinexpr = NULL;
   sepadata->sbilinvars = 0;
   sepadata->nbilinvars = 0;

   /* get all bilinear terms from the expression constraint handler */
   bilinterms = SCIPgetConsExprBilinTerms(sepadata->conshdlr);

   /* store the information of all variables that appear bilinearly */
   for( i = 0; i < nbilinterms; ++i )
   {
      assert(bilinterms[i].x != NULL);
      assert(bilinterms[i].y != NULL);
      assert(bilinterms[i].nlockspos + bilinterms[i].nlocksneg > 0);

      /* skip bilinear term if it does not have an auxiliary variable */
      if( bilinterms[i].aux.var == NULL )
         continue;

      /* if only initial rows are requested, skip products that contain at least one auxiliary variable */
      if( sepadata->onlyinitial && (SCIPvarIsRelaxationOnly(bilinterms[i].x) ||
          SCIPvarIsRelaxationOnly(bilinterms[i].y)) )
         continue;

      SCIP_CALL( addProductVars(scip, sepadata, bilinterms[i].x, bilinterms[i].y, varmap,
            bilinterms[i].nlockspos + bilinterms[i].nlocksneg) );
   }

   if( sepadata->detecthidden )
   {
      int oldnterms = SCIPgetConsExprNBilinTerms(sepadata->conshdlr);

      SCIP_CALL( detectHiddenProducts(scip, sepadata, varmap) );

      if( SCIPgetConsExprNBilinTerms(sepadata->conshdlr) - oldnterms > 0 )
      {
         SCIPinfoMessage(scip, NULL, "\nFound hidden products");
         SCIPinfoMessage(scip, NULL, "\nNumber of hidden products: %d",
                                      SCIPgetConsExprNBilinTerms(sepadata->conshdlr) - oldnterms);
      }
   }

   SCIPhashmapFree(&varmap);
   sepadata->nbilinterms = SCIPgetConsExprNBilinTerms(sepadata->conshdlr);

   if( sepadata->nbilinterms == 0 )
   {
      SCIPhashmapFree(&sepadata->bilinvarsmap);
      return SCIP_OKAY;
   }

   if( sepadata->detecthidden )
   {
      /* update bilinterms, as detectHiddenProducts might have found new terms */
      bilinterms = SCIPgetConsExprBilinTerms(sepadata->conshdlr);
   }

   /* mark positions of equality relations */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sepadata->eqlinexpr, sepadata->nbilinterms) );

   for( i = 0; i < sepadata->nbilinterms; ++i )
   {
      int j;

      sepadata->eqlinexpr[i] = -1;
      for( j = 0; j < bilinterms[i].nauxexprs; ++j )
      {
         assert(bilinterms[i].aux.exprs[j] != NULL);

         if( bilinterms[i].aux.exprs[j]->underestimate && bilinterms[i].aux.exprs[j]->overestimate )
         {
            sepadata->eqlinexpr[i] = j;
            break;
         }
      }
   }

   /* sort maxnumber of variables according to their occurrences */
   SCIPselectDownPtrPtr((void**) sepadata->bilinvardatas, (void**) sepadata->varssorted, bilinVarDataComp,
         sepadata->maxusedvars, sepadata->nbilinvars);

   /* capture all variables */
   for( i = 0; i < sepadata->nbilinvars; ++i )
   {
      assert(sepadata->varssorted[i] != NULL);
      SCIP_CALL( SCIPcaptureVar(scip, sepadata->varssorted[i]) );
   }

   /* mark that separation data has been created */
   sepadata->iscreated = TRUE;
   sepadata->isinitialround = TRUE;

   if( SCIPgetConsExprNBilinTerms(sepadata->conshdlr) > 0 )
      SCIPinfoMessage(scip, NULL, "\nFound bilinear terms\n");
   else
      SCIPinfoMessage(scip, NULL, "\nNo bilinear terms");

   return SCIP_OKAY;
}

/** get the positions of the most violated linear under- and overestimators for all products */
static
void getBestEstimators(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_SOL*             sol,                /**< solution at which to evaluate the expressions */
   int*                  bestunderestimators,/**< array of indices of best underestimators for each term */
   int*                  bestoverestimators  /**< array of indices of best overestimators for each term */
)
{
   SCIP_Real prodval;
   SCIP_Real linval;
   SCIP_Real prodviol;
   SCIP_Real viol_below;
   SCIP_Real viol_above;
   int i;
   int j;
   SCIP_CONSEXPR_BILINTERM* terms;

   assert(bestunderestimators != NULL);
   assert(bestoverestimators != NULL);

   terms = SCIPgetConsExprBilinTerms(sepadata->conshdlr);

   for( j = 0; j < SCIPgetConsExprNBilinTerms(sepadata->conshdlr); ++j )
   {
      viol_below = -SCIPinfinity(scip);
      viol_above = -SCIPinfinity(scip);

      /* evaluate the product expression */
      prodval = SCIPgetSolVal(scip, sol, terms[j].x) * SCIPgetSolVal(scip, sol, terms[j].y);

      bestunderestimators[j] = -1;
      bestoverestimators[j] = -1;

      /* look for the best under- and overestimator, store their positions */

      /* if there are any auxexprs, look there */
      for( i = 0; i < terms[j].nauxexprs; ++i )
      {
         linval = SCIPevalConsExprBilinAuxExpr(scip, terms[j].x, terms[j].y, terms[j].aux.exprs[i], sol);
         prodviol = linval - prodval;

         if( terms[j].aux.exprs[i]->underestimate && prodviol > viol_below )
         {
            viol_below = prodviol;
            bestunderestimators[j] = i;
         }
         if( terms[j].aux.exprs[i]->overestimate && -prodviol > viol_above )
         {
            viol_above = -prodviol;
            bestoverestimators[j] = i;
         }

         /* TODO also used to find equality relations here - move this elsewhere */
      }

      /* if the term has a plain auxvar, it will be treated differently - do nothing here */
   }
}

/** tests if a row contains too many unknown bilinear terms w.r.t. the parameters */
static
SCIP_RETCODE isAcceptableRow(
   SCIP_SEPADATA*        sepadata,           /**< separation data */
   SCIP_ROW*             row,                /**< the row to be tested */
   SCIP_VAR*             var,                /**< the variable that is to be multiplied with row */
   int*                  currentnunknown,    /**< number of unknown terms in current row */
   SCIP_Bool*            acceptable          /**< buffer to store the result */
   )
{
   int i;
   int idx;
   SCIP_CONSEXPR_BILINTERM* terms;

   assert(row != NULL);
   assert(var != NULL);

   *currentnunknown = 0;
   terms = SCIPgetConsExprBilinTerms(sepadata->conshdlr);

   for( i = 0; (i < SCIProwGetNNonz(row)) && (sepadata->maxunknownterms < 0 || *currentnunknown <= sepadata->maxunknownterms); ++i )
   {
      idx = SCIPgetConsExprBilinTermIdx(sepadata->conshdlr, var, SCIPcolGetVar(SCIProwGetCols(row)[i]));

      /* if the product hasn't been found, no linearisations for it are known */
      if( idx < 0 )
      {
         ++(*currentnunknown);
         continue;
      }

      /* known terms are only those that have equality estimators */
      if( sepadata->eqlinexpr[idx] == -1 && !(terms[idx].nauxexprs == 0 && terms[idx].aux.var != NULL) )
      {
         ++(*currentnunknown);
      }
   }

   *acceptable = sepadata->maxunknownterms < 0 || *currentnunknown <= sepadata->maxunknownterms;

   return SCIP_OKAY;
}

/** computes coefficients of an auxiliary expression (coef*auxexpr) for a product term */
static
void addAuxexprToRow(
   SCIP_VAR*             var1,               /**< first product variable */
   SCIP_VAR*             var2,               /**< second product variable */
   SCIP_CONSEXPR_AUXEXPR* auxexpr,           /**< auxiliary expression to be added */
   SCIP_Real             coef,               /**< coefficient of the linearisation */
   SCIP_Real*            coefaux,            /**< coefficient of the auxiliary variable */
   SCIP_Real*            coef1,              /**< coefficient of the first variable */
   SCIP_Real*            coef2,              /**< coefficient of the second variable */
   SCIP_Real*            finalside           /**< buffer that stores the side of the cut */
)
{
   assert(auxexpr->auxvar != NULL);

   *coefaux += auxexpr->coefs[0] * coef;
   if( SCIPvarCompare(var1, var2) < 1 )
   {
      *coef1 += auxexpr->coefs[1] * coef;
      *coef2 += auxexpr->coefs[2] * coef;
   }
   else
   {
      *coef1 += auxexpr->coefs[2] * coef;
      *coef2 += auxexpr->coefs[1] * coef;
   }
   *finalside += coef * auxexpr->cst;
}

/* add a linearisation of term coef*colvar*var to cut
 *
 * adds the linear term involving colvar to cut and updates coefvar and finalside
 */
static
SCIP_RETCODE addRltTerm(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_SOL*             sol,                /**< the point to be separated (can be NULL) */
   int*                  bestunderest,       /**< positions of most violated underestimators for each product term */
   int*                  bestoverest,        /**< positions of most violated overestimators for each product term */
   SCIP_ROW*             cut,                /**< cut to which the term is to be added */
   SCIP_VAR*             var,                /**< multiplier variable */
   SCIP_VAR*             colvar,             /**< row variable to be multiplied */
   SCIP_Real             coef,               /**< coefficient of the bilinear term */
   SCIP_Bool             uselb,              /**< whether we multiply with (var - lb) or (ub - var) */
   SCIP_Bool             uselhs,             /**< whether to create a cut for the lhs or rhs */
   SCIP_Bool             local,              /**< whether local or global cuts should be computed */
   SCIP_Bool             computeEqCut,       /**< whether conditions are fulfilled to compute equality cuts */
   SCIP_Real*            coefvar,            /**< coefficient of var */
   SCIP_Real*            finalside,          /**< buffer to store the left or right hand side of cut */
   SCIP_Bool*            success             /**< buffer to store whether cut was created successfully */
   )
{
   SCIP_Real lbvar;
   SCIP_Real ubvar;
   SCIP_Real refpointvar;
   SCIP_Real signfactor;
   SCIP_Real boundfactor;
   SCIP_Real coefauxvar;
   SCIP_Real coefcolvar;
   SCIP_Real coefterm;
   int linpos;
   int idx;
   SCIP_CONSEXPR_BILINTERM* terms;
   SCIP_VAR* auxvar;

   terms = SCIPgetConsExprBilinTerms(sepadata->conshdlr);

   lbvar = local ? SCIPvarGetLbLocal(var) : SCIPvarGetLbGlobal(var);
   ubvar = local ? SCIPvarGetUbLocal(var) : SCIPvarGetUbGlobal(var);

   refpointvar = MAX(lbvar, MIN(ubvar, SCIPgetSolVal(scip, sol, var))); /*lint !e666*/

   signfactor = (uselb ? 1.0 : -1.0);
   boundfactor = (uselb ? -lbvar : ubvar);

   coefterm = coef * signfactor;
   coefcolvar = coef * boundfactor;
   coefauxvar = 0.0;
   auxvar = NULL;

   assert(!SCIPisInfinity(scip, REALABS(coefterm)));

   idx = SCIPgetConsExprBilinTermIdx(sepadata->conshdlr, var, colvar);
   linpos = -1;

   /* for an implicit term, get the position of the best estimator */
   if( idx >= 0 && terms[idx].nauxexprs > 0 )
   {
      if( computeEqCut )
      { /* use an equality linearisation (which should exist for computeEqCut to be TRUE) */
         assert(sepadata->eqlinexpr[idx] >= 0);
         linpos = sepadata->eqlinexpr[idx];
      }
      else if( (uselhs && coefterm > 0) || (!uselhs && coefterm < 0) )
      { /* use an overestimator */
         linpos = bestoverest[idx];
      }
      else
      { /* use an underestimator */
         linpos = bestunderest[idx];
      }
   }

   /* if the term is implicit and a suitable linearisation for it exists,
    * add the linearisation to the cut with the previous coefficient
    */
   if( linpos >= 0 )
   {
      SCIPdebugMsg(scip, "linearisation for %s and %s found, will be added to cut:\n",
                          SCIPvarGetName(colvar), SCIPvarGetName(var));
      addAuxexprToRow(var, colvar, terms[idx].aux.exprs[linpos], coefterm, &coefauxvar, coefvar, &coefcolvar, finalside);
      auxvar = terms[idx].aux.exprs[linpos]->auxvar;
   }
   /* for an existing term, use the auxvar if there is one */
   else if( idx >= 0 && terms[idx].nauxexprs == 0 && terms[idx].aux.var != NULL )
   {
      SCIPdebugMsg(scip, "auxvar for %s and %s found, will be added to cut:\n",
                   SCIPvarGetName(colvar), SCIPvarGetName(var));
      coefauxvar += coefterm;
      auxvar = terms[idx].aux.var;
   }
   /* otherwise, use clique information or the McCormick estimator in place of the bilinear term */
   else if( colvar != var )
   {
      SCIP_Bool found_clique = FALSE;
      SCIP_Real lbcolvar = local ? SCIPvarGetLbLocal(colvar) : SCIPvarGetLbGlobal(colvar);
      SCIP_Real ubcolvar = local ? SCIPvarGetUbLocal(colvar) : SCIPvarGetUbGlobal(colvar);
      SCIP_Real refpointcolvar = MAX(lbcolvar, MIN(ubcolvar, SCIPgetSolVal(scip, sol, colvar))); /*lint !e666*/

      assert(!computeEqCut);

      if( REALABS(lbcolvar) > MAXVARBOUND || REALABS(ubcolvar) > MAXVARBOUND )
      {
         *success = FALSE;
         return SCIP_OKAY;
      }

      SCIPdebugMsg(scip, "auxvar for %s and %s not found, will linearise the product\n", SCIPvarGetName(colvar), SCIPvarGetName(var));

      /* if both variables are binary, check if they are contained together in some clique */
      if( SCIPvarGetType(var) == SCIP_VARTYPE_BINARY &&  SCIPvarGetType(colvar) == SCIP_VARTYPE_BINARY )
      {
         int c;
         SCIP_CLIQUE** varcliques;

         varcliques = SCIPvarGetCliques(var, TRUE);

         /* look through cliques containing var */
         for( c = 0; c < SCIPvarGetNCliques(var, TRUE); ++c )
         {
            if( SCIPcliqueHasVar(varcliques[c], colvar, TRUE) ) /* var + colvar <= 1 => var*colvar = 0 */
            {
               /* product is zero, add nothing */
               found_clique = TRUE;
               break;
            }

            if( SCIPcliqueHasVar(varcliques[c], colvar, FALSE) ) /* var + (1-colvar) <= 1 => var*colvar = var */
            {
               *coefvar += coefterm;
               found_clique = TRUE;
               break;
            }
         }

         if( !found_clique )
         {
            varcliques = SCIPvarGetCliques(var, FALSE);

            /* look through cliques containing complement of var */
            for( c = 0; c < SCIPvarGetNCliques(var, FALSE); ++c )
            {
               if( SCIPcliqueHasVar(varcliques[c], colvar, TRUE) ) /* (1-var) + colvar <= 1 => var*colvar = colvar */
               {
                  coefcolvar += coefterm;
                  found_clique = TRUE;
                  break;
               }

               if( SCIPcliqueHasVar(varcliques[c], colvar, FALSE) ) /* (1-var) + (1-colvar) <= 1 => var*colvar = var + colvar - 1 */
               {
                  *coefvar += coefterm;
                  coefcolvar += coefterm;
                  *finalside -= coefterm;
                  found_clique = TRUE;
                  break;
               }
            }
         }
      }

      if( !found_clique )
      {
         SCIPdebugMsg(scip, "clique for %s and %s not found or at least one of them is not binary, will use McCormick\n", SCIPvarGetName(colvar), SCIPvarGetName(var));
         SCIPaddBilinMcCormick(scip, coefterm, lbvar, ubvar, refpointvar, lbcolvar,
            ubcolvar, refpointcolvar, uselhs, coefvar, &coefcolvar, finalside, success);
         if( !*success )
            return SCIP_OKAY;
      }
   }

   /* or, if it's a quadratic term, use a secant for overestimation and a gradient for underestimation */
   else
   {
      SCIPdebugMsg(scip, "auxvar for %s^2 not found, will use gradient and secant estimators\n", SCIPvarGetName(colvar));

      assert(!computeEqCut);

      /* for a binary var, var^2 = var */
      if( SCIPvarGetType(var) == SCIP_VARTYPE_BINARY )
      {
         *coefvar += coefterm;
      }
      else
      {
         /* depending on over-/underestimation and the sign of the column variable, compute secant or tangent */
         if( (uselhs && coefterm > 0.0) || (!uselhs && coefterm < 0.0) )
            SCIPaddSquareSecant(scip, coefterm, lbvar, ubvar, coefvar, finalside, success);
         else
            SCIPaddSquareLinearization(scip, coefterm, refpointvar, SCIPvarIsIntegral(var), coefvar, finalside, success);

         if( !*success )
            return SCIP_OKAY;
      }
   }

   /* add the auxiliary variable if its coefficient is nonzero */
   if( !SCIPisZero(scip, coefauxvar) )
   {
      assert(auxvar != NULL);
      SCIP_CALL( SCIPaddVarToRow(scip, cut, auxvar, coefauxvar) );
   }

   /* add the linear term for this column */
   if( colvar != var )
   {
      assert(!SCIPisInfinity(scip, REALABS(coefcolvar)));
      SCIP_CALL( SCIPaddVarToRow(scip, cut, colvar, coefcolvar) );
   }
   else
      *coefvar += coefcolvar;

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
   SCIP_SOL*             sol,                /**< the point to be separated (can be NULL) */
   int*                  bestunderest,       /**< positions of most violated underestimators for each product term */
   int*                  bestoverest,        /**< positions of most violated overestimators for each product term */
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
   SCIP_Real coefvar;
   SCIP_Real constside;
   SCIP_Real finalside;
   int i;
   char name[SCIP_MAXSTRLEN];

   assert(sepadata != NULL);
   assert(cut != NULL);
   assert(row != NULL);
   assert(var != NULL);
   assert(success != NULL);
   assert(!computeEqCut || SCIPisEQ(scip, SCIProwGetLhs(row), SCIProwGetRhs(row)));

   *cut = NULL;

   /* get data for given variable */
   if( computeEqCut )
   {
      lbvar = 0.0;
      ubvar = 0.0;
   }
   else
   {
      lbvar = local ? SCIPvarGetLbLocal(var) : SCIPvarGetLbGlobal(var);
      ubvar = local ? SCIPvarGetUbLocal(var) : SCIPvarGetUbGlobal(var);
   }

   constside = uselhs ? SCIProwGetLhs(row) : SCIProwGetRhs(row);

   /* if the bounds are too large or the respective side is infinity, skip this cut */
   if( (uselb && REALABS(lbvar) > MAXVARBOUND) || (!uselb && REALABS(ubvar) > MAXVARBOUND)
      || SCIPisInfinity(scip, REALABS(constside)) )
   {
      SCIPdebugMsg(scip, "cut generation for row %s, %s and variable %s with its %s %g not possible\n",
         SCIProwGetName(row), uselhs ? "lhs" : "rhs", SCIPvarGetName(var),
         uselb ? "lower bound" : "upper bound", uselb ? lbvar : ubvar);

      if( REALABS(lbvar) > MAXVARBOUND )
         SCIPdebugMsg(scip, " because of lower bound\n");
      if( REALABS(ubvar) > MAXVARBOUND )
         SCIPdebugMsg(scip, " because of upper bound\n");
      if( SCIPisInfinity(scip, REALABS(constside)) )
         SCIPdebugMsg(scip, " because of side %g\n", constside);

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
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "rlt_cut_%s_%s_%s_%s_%d", SCIProwGetName(row), uselhs ? "lhs" : "rhs",
                       SCIPvarGetName(var), uselb ? "lb" : "ub", SCIPgetNLPs(scip));
   SCIP_CALL( SCIPcreateEmptyRowSepa(scip, cut, sepa, name, -SCIPinfinity(scip), SCIPinfinity(scip),
         SCIPgetDepth(scip) > 0 && local, FALSE, FALSE) );

   SCIP_CALL( SCIPcacheRowExtensions(scip, *cut) );

   /* iterate over all variables in the row and add the corresponding terms to the cuts */
   for( i = 0; i < SCIProwGetNNonz(row); ++i )
   {
      SCIP_VAR* colvar;
      colvar = SCIPcolGetVar(SCIProwGetCols(row)[i]);
      SCIP_CALL( addRltTerm(scip, sepadata, sol, bestunderest, bestoverest, *cut, var, colvar, SCIProwGetVals(row)[i],
            uselb, uselhs, local, computeEqCut, &coefvar, &finalside, success) );
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
   assert(!SCIPisInfinity(scip, REALABS(coefvar)));
   SCIP_CALL( SCIPaddVarToRow(scip, *cut, var, coefvar) );

   SCIP_CALL( SCIPflushRowExtensions(scip, *cut) );

   assert(!SCIPisInfinity(scip, REALABS(finalside)));
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

/* TODO make one function out of this and the above */
/** creates the RLT cuts formed by multiplying a given projected row with (x - lb) or (ub - x)
 *
 * in detail:
 * -The row is multiplied either with (x - lb(x)) or with (ub(x) - x), depending on parameter uselb.
 * -The cut is computed either for lhs or rhs, depending on parameter uselhs.
 * -Terms for which no auxiliary variable exists are replaced by either McCormick, secants, or linearization cuts
 */
static
SCIP_RETCODE computeProjRltCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SEPADATA*        sepadata,           /**< separation data */
   SCIP_ROW**            cut,                /**< buffer to store the cut */
   RLT_SIMPLEROW*        projrows,           /**< projected rows */
   int                   idx,                /**< index of the row that is used for the rlt cut */
   SCIP_SOL*             sol,                /**< the point to be separated (can be NULL) */
   int*                  bestunderest,       /**< positions of most violated underestimators for each product term */
   int*                  bestoverest,        /**< positions of most violated overestimators for each product term */
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
   SCIP_Real coefvar;
   SCIP_Real constside;
   SCIP_Real finalside;
   int i;
   char name[SCIP_MAXSTRLEN];

   assert(sepadata != NULL);
   assert(cut != NULL);
   assert(projrows != NULL);
   assert(var != NULL);
   assert(success != NULL);
   assert(!computeEqCut || SCIPisEQ(scip, projrows[idx].lhs, projrows[idx].rhs));

   *cut = NULL;

   /* get data for given variable */
   lbvar = local ? SCIPvarGetLbLocal(var) : SCIPvarGetLbGlobal(var);
   ubvar = local ? SCIPvarGetUbLocal(var) : SCIPvarGetUbGlobal(var);
   constside = uselhs ? projrows[idx].lhs : projrows[idx].rhs;

   /* if the bounds are too large or the respective side is infinity, skip this cut */
   if( REALABS(lbvar) > MAXVARBOUND || REALABS(ubvar) > MAXVARBOUND || SCIPisInfinity(scip, REALABS(constside)) )
   {
      SCIPdebugMsg(scip, "cut generation for projected row %d, %s and variable %s with its %s %g not possible\n",
                   idx, uselhs ? "lhs" : "rhs", SCIPvarGetName(var),
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
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "rlt_proj_cut_%d_%s_%s_%s_%d", idx, uselhs ? "lhs" : "rhs",
                       SCIPvarGetName(var), uselb ? "lb" : "ub", SCIPgetNLPs(scip));
   SCIP_CALL( SCIPcreateEmptyRowSepa(scip, cut, sepa, "rlt_cut", -SCIPinfinity(scip), SCIPinfinity(scip),
         TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPcacheRowExtensions(scip, *cut) );

   /* iterate over all variables in the row and add the corresponding terms to the cuts */
   for( i = 0; i < projrows[idx].nNonz; ++i )
   {
      SCIP_CALL( addRltTerm(scip, sepadata, sol, bestunderest, bestoverest, *cut, var, projrows[idx].vars[i],
            projrows[idx].coefs[i], uselb, uselhs, local, computeEqCut, &coefvar, &finalside, success) );
   }

   if( REALABS(finalside) > MAXVARBOUND )
   {
      *success = FALSE;
      return SCIP_OKAY;
   }

   /* multiply (x-lb) or (ub -x) with the lhs and rhs of the row */
   coefvar += signfactor * (projrows[idx].cst - constside);
   finalside = boundfactor * (constside - projrows[idx].cst) - finalside;

   /* set the coefficient of var and the constant side */
   assert(!SCIPisInfinity(scip, REALABS(coefvar)));
   SCIP_CALL( SCIPaddVarToRow(scip, *cut, var, coefvar) );

   SCIP_CALL( SCIPflushRowExtensions(scip, *cut) );

   assert(!SCIPisInfinity(scip, REALABS(finalside)));
   if( uselhs || computeEqCut )
   {
      SCIP_CALL( SCIPchgRowLhs(scip, *cut, finalside) );
   }
   if( !uselhs || computeEqCut )
   {
      SCIP_CALL( SCIPchgRowRhs(scip, *cut, finalside) );
   }

   SCIPdebugMsg(scip, "projected cut was generated successfully:\n");
#ifdef SCIP_DEBUG
   SCIP_CALL( SCIPprintRow(scip, *cut, NULL) );
#endif

   return SCIP_OKAY;
}

/** store a row projected by fixing all variables that are at bound at sol; the result is a simplified row */
static
SCIP_RETCODE createProjRow(
   SCIP*                 scip,               /**< SCIP data structure */
   RLT_SIMPLEROW*        simplerow,          /**< pointer to the simplified row */
   SCIP_ROW*             row,                /**< row to be projected */
   SCIP_SOL*             sol,                /**< the point to be separated (can be NULL) */
   SCIP_Bool             local               /**< whether local bounds should be checked */
  )
{
   int i;
   SCIP_VAR* var;
   SCIP_Real val;
   SCIP_Real vlb;
   SCIP_Real vub;

   assert(simplerow != NULL);

   simplerow->nNonz = 0;
   simplerow->size = 0;
   simplerow->vars = NULL;
   simplerow->coefs = NULL;
   simplerow->lhs = SCIProwGetLhs(row);
   simplerow->rhs = SCIProwGetRhs(row);
   simplerow->cst = SCIProwGetConstant(row);

   for( i = 0; i < SCIProwGetNNonz(row); ++i )
   {
      var = SCIPcolGetVar(SCIProwGetCols(row)[i]);
      val = SCIPgetSolVal(scip, sol, var);
      vlb = local ? SCIPvarGetLbLocal(var) : SCIPvarGetLbGlobal(var);
      vub = local ? SCIPvarGetUbLocal(var) : SCIPvarGetUbGlobal(var);
      if( SCIPisEQ(scip, vlb, val) || SCIPisEQ(scip, vub, val) )
      {
         /* if we are projecting and the var is at bound, add var as a constant to simplerow */
         if( !SCIPisInfinity(scip, -simplerow->lhs) )
            simplerow->lhs -= SCIProwGetVals(row)[i]*val;
         if( !SCIPisInfinity(scip, simplerow->rhs) )
            simplerow->rhs -= SCIProwGetVals(row)[i]*val;
      }
      else
      {
         if( simplerow->nNonz + 1 > simplerow->size )
         {
            int newsize;

            newsize = SCIPcalcMemGrowSize(scip, simplerow->nNonz + 1);
            SCIP_CALL( SCIPreallocBufferArray(scip, &simplerow->coefs, newsize) );
            SCIP_CALL( SCIPreallocBufferArray(scip, &simplerow->vars, newsize) );
            simplerow->size = newsize;
         }

         /* add the term to simplerow */
         simplerow->vars[simplerow->nNonz] = var;
         simplerow->coefs[simplerow->nNonz] = SCIProwGetVals(row)[i];
         ++(simplerow->nNonz);
      }
   }

   return SCIP_OKAY;
}

/** free the projected row */
static
void freeProjRow(
   SCIP*                 scip,               /**< SCIP data structure */
   RLT_SIMPLEROW*        simplerow           /**< simplified row to be freed */
   )
{
   assert(simplerow != NULL);

   if( simplerow->size > 0 )
   {
      assert(simplerow->vars != NULL);
      assert(simplerow->coefs != NULL);

      SCIPfreeBufferArray(scip, &simplerow->vars);
      SCIPfreeBufferArray(scip, &simplerow->coefs);
   }
}

/** creates the projected problem
 *
 *  All variables that are at their bounds at the current solution are added
 *  to left and/or right hand sides as constant values.
 */
static
SCIP_RETCODE createProjRows(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            rows,               /**< problem rows */
   int                   nrows,              /**< number of rows */
   SCIP_SOL*             sol,                /**< the point to be separated (can be NULL) */
   RLT_SIMPLEROW**       projrows,           /**< the projected rows to be filled */
   SCIP_Bool             local,              /**< are local cuts allowed? */
   SCIP_Bool*            allcst              /**< buffer to store whether all projected rows have only constants */
   )
{
   int i;

   assert(scip != NULL);
   assert(rows != NULL);
   assert(projrows != NULL);
   assert(allcst != NULL);

   *allcst = TRUE;
   SCIP_CALL( SCIPallocBufferArray(scip, projrows, nrows) );

   for( i = 0; i < nrows; ++i )
   {
      /* get a simplified and projected row */
      SCIP_CALL( createProjRow(scip, &(*projrows)[i], rows[i], sol, local) );
      if( (*projrows)[i].nNonz > 0 )
      {
         *allcst = FALSE;
      }
   }

   return SCIP_OKAY;
}

#ifdef SCIP_DEBUG
/* prints the projected LP */
static
void printProjRows(
   SCIP*                 scip,               /**< SCIP data structure */
   RLT_SIMPLEROW*        projrows,           /**< the projected rows */
   int                   nrows,              /**< number of projected rows */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   int i;
   int j;

   assert(projrows != NULL);

   for( i = 0; i < nrows; ++i )
   {
      SCIPinfoMessage(scip, file, "\n%s: ", "proj_row[%d]", i);
      if( !SCIPisInfinity(scip, -projrows[i].lhs) )
         SCIPinfoMessage(scip, file, "%.15g <= ", projrows[i].lhs);
      for( j = 0; j < projrows[i].nNonz; ++j )
      {
         if( j == 0 )
         {
            if( projrows[i].coefs[j] < 0 )
               SCIPinfoMessage(scip, file, "-");
         }
         else
         {
            if( projrows[i].coefs[j] < 0 )
               SCIPinfoMessage(scip, file, " - ");
            else
               SCIPinfoMessage(scip, file, " + ");
         }

         if( projrows[i].coefs[j] != 1.0 )
            SCIPinfoMessage(scip, file, "%.15g*", REALABS(projrows[i].coefs[j]));
         SCIPinfoMessage(scip, file, "<%s>", SCIPvarGetName(projrows[i].vars[j]));
      }
      if( projrows[i].cst > 0 )
         SCIPinfoMessage(scip, file, " + %.15g", projrows[i].cst);
      else if( projrows[i].cst < 0 )
         SCIPinfoMessage(scip, file, " - %.15g", REALABS(projrows[i].cst));

      if( !SCIPisInfinity(scip, projrows[i].rhs) )
         SCIPinfoMessage(scip, file, " <= %.15g", projrows[i].rhs);
   }
   SCIPinfoMessage(scip, file, "\n");
}
#endif

/** frees the projected rows */
static
void freeProjRows(
   SCIP*                 scip,               /**< SCIP data structure */
   RLT_SIMPLEROW**       projrows,           /**< the projected LP */
   int                   nrows               /**< number of rows in projrows */
   )
{
   int i;

   for( i = 0; i < nrows; ++i )
   {
      freeProjRow(scip, &(*projrows)[i]);
   }

   SCIPfreeBufferArray(scip, projrows);
}

/* mark a row for rlt cut selection
 *
 * depending on the sign of value and row inequality type, set the mark to:
 * 1 - cuts for axy < aw case,
 * 2 - cuts for axy > aw case,
 * 3 - cuts for both cases
 */
static
void addRowMark(
   int                   ridx,               /**< row index */
   SCIP_Real             coef,               /**< ai*(w - xy) */
   SCIP_Real             prod_viol_below,    /**< violation of the product from below (0 or positive) */
   SCIP_Real             prod_viol_above,    /**< violation of the product from above (0 or positive) */
   int*                  row_idcs,           /**< sparse array with indices of marked rows */
   int*                  row_marks,          /**< sparse array to store the marks */
   int*                  nmarked             /**< number of marked rows */
)
{
   int newmark;
   int pos;
   SCIP_Bool exists;

   assert(coef != 0.0);

   if( (coef > 0.0 && prod_viol_below > 0.0) || (coef < 0.0 && prod_viol_above > 0.0 ) )
      newmark = 1; /* axy < aw case */
   else newmark = 2; /* axy > aw case */

   /* find row idx in row_idcs */
   exists = SCIPsortedvecFindInt(row_idcs, ridx, *nmarked, &pos);

   if( exists )
   {
      /* we found the row index: update the mark at pos1 */
      if( (newmark == 1 && row_marks[pos] == 2) || (newmark == 2 && row_marks[pos] == 1) )
      {
         row_marks[pos] = 3;
      }
   }
   else /* the given row index does not yet exist in row_idcs */
   {
      int i;

      /* insert row index at the correct position */
      for( i = *nmarked; i > pos; --i )
      {
         row_idcs[i] = row_idcs[i-1];
         row_marks[i] = row_marks[i-1];
      }
      row_idcs[pos] = ridx;
      row_marks[pos] = newmark;
      (*nmarked)++;
   }
}

/* mark all rows that should be multiplied by xj */
static
SCIP_RETCODE markRowsXj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_SOL*             sol,                /**< point to be separated (can be NULL) */
   int                   j,                  /**< index of the multiplier variable in sepadata */
   SCIP_Bool             local,              /**< are local cuts allowed? */
   SCIP_HASHMAP*         row_to_pos,         /**< hashmap linking row indices to positions in array */
   int*                  bestunderest,       /**< positions of most violated underestimators for each product term */
   int*                  bestoverest,        /**< positions of most violated overestimators for each product term */
   int*                  row_marks,          /**< sparse array storing the row marks */
   int*                  row_idcs,           /**< sparse array storing the marked row positions */
   int*                  nmarked             /**< number of marked rows */
)
{
   int i;
   int idx;
   int ncolrows;
   int r;
   int ridx;
   SCIP_VAR* xi;
   SCIP_VAR* xj;
   SCIP_Real vlb;
   SCIP_Real vub;
   SCIP_Real vali;
   SCIP_Real valj;
   SCIP_Real a;
   SCIP_COL* coli;
   SCIP_Real* colvals;
   SCIP_Real viol_below;
   SCIP_Real viol_above;
   SCIP_ROW** colrows;
   SCIP_CONSEXPR_BILINTERM* terms;
   BILINVARDATA* bilinvardata;

   *nmarked = 0;

   xj = sepadata->varssorted[j];
   assert(xj != NULL);

   valj = SCIPgetSolVal(scip, sol, xj);
   vlb = local ? SCIPvarGetLbLocal(xj) : SCIPvarGetLbGlobal(xj);
   vub = local ? SCIPvarGetUbLocal(xj) : SCIPvarGetUbGlobal(xj);

   if( sepadata->useprojection && (SCIPisEQ(scip, vlb, valj) || SCIPisEQ(scip, vub, valj)) )
   {
      /* we don't want to multiply by variables that are at bound */
      SCIPdebugMsg(scip, "Rejected multiplier %s in [%g,%g] because it is at bound (current value %g)\n", SCIPvarGetName(xj), vlb, vub, valj);
      return SCIP_OKAY;
   }

   terms = SCIPgetConsExprBilinTerms(conshdlr);
   bilinvardata = sepadata->bilinvardatas[j];

   /* for each var which appears in a bilinear product together with xj, mark rows */
   for( i = 0; i < bilinvardata->nvarbilinvars; ++i )
   {
      xi = bilinvardata->varbilinvars[i];

      if( SCIPvarGetStatus(xi) != SCIP_VARSTATUS_COLUMN )
         continue;

      vali = SCIPgetSolVal(scip, sol, xi);
      vlb = local ? SCIPvarGetLbLocal(xi) : SCIPvarGetLbGlobal(xi);
      vub = local ? SCIPvarGetUbLocal(xi) : SCIPvarGetUbGlobal(xi);

      if( sepadata->useprojection && (SCIPisEQ(scip, vlb, vali) || SCIPisEQ(scip, vub, vali)) ) /* we aren't interested in products with variables that are at bound */
         continue;

      /* get the index of the bilinear product */
      idx = SCIPgetConsExprBilinTermIdx(conshdlr, xj, xi);
      assert(idx >= 0 && idx < SCIPgetConsExprNBilinTerms(conshdlr));

      if( !sepadata->hiddenrlt && !terms[idx].existing )
         continue;

      /* use the most violated under- and overestimators for this product;
       * if equality cuts are computed, we might end up using a different linearisation;
       * so this is an optimistic (i.e. taking the largest possible violation) estimation
       */
      if( bestunderest == NULL || bestunderest[idx] == -1 )
      {
         if( terms[idx].nauxexprs == 0 && terms[idx].aux.var != NULL )
         {
            assert(terms[idx].existing);
            viol_below = SCIPgetSolVal(scip, sol, terms[idx].aux.var) - valj * vali;
         }
         else
         {
            assert(bestunderest != NULL);
            viol_below = 0.0;
         }
      }
      else
      {
         assert(bestunderest[idx] >= 0 && bestunderest[idx] < terms[idx].nauxexprs);
         viol_below = SCIPevalConsExprBilinAuxExpr(scip, terms[idx].x, terms[idx].y,
               terms[idx].aux.exprs[bestunderest[idx]], sol) - valj * vali;
      }

      if( bestoverest == NULL || bestoverest[idx] == -1 )
      {
         if( terms[idx].nauxexprs == 0 && terms[idx].aux.var != NULL )
         {
            assert(terms[idx].existing);
            viol_above = valj * vali - SCIPgetSolVal(scip, sol, terms[idx].aux.var);
         }
         else
         {
            assert(bestoverest != NULL);
            viol_above = 0.0;
         }
      }
      else
      {
         assert(bestoverest[idx] >= 0 && bestoverest[idx] < terms[idx].nauxexprs);
         viol_above = valj * vali - SCIPevalConsExprBilinAuxExpr(scip, terms[idx].x, terms[idx].y,
               terms[idx].aux.exprs[bestoverest[idx]], sol);
      }

      SCIPdebugMsg(scip, "prodval = %g, prod viol below = %g, above = %g\n", valj * vali, viol_below, viol_above);

      /* we are interested only in violated product relations */
      if( SCIPisFeasLE(scip, viol_below, 0.0) && SCIPisFeasLE(scip, viol_above, 0.0) )
      {
         SCIPdebugMsg(scip, "the product for vars %s, %s is not violated\n", SCIPvarGetName(xj), SCIPvarGetName(xi));
         continue;
      }

      /* get the column of xi */
      coli = SCIPvarGetCol(xi);
      colvals = SCIPcolGetVals(coli);
      ncolrows = SCIPcolGetNNonz(coli);
      colrows = SCIPcolGetRows(coli);

      SCIPdebugMsg(scip, "marking rows for xj, xi = %s, %s\n", SCIPvarGetName(xj), SCIPvarGetName(xi));

      /* mark the rows */
      for( r = 0; r < ncolrows; ++r )
      {
         ridx = SCIProwGetIndex(colrows[r]);

         if( !SCIPhashmapExists(row_to_pos, (void*)(size_t)ridx) )
            continue; /* if row index is not in row_to_pos, it means that storeSuitableRows decided to ignore this row */

         a = colvals[r];
         if( a == 0.0 )
            continue;

         SCIPdebugMsg(scip, "Marking row %d\n", ridx);
         addRowMark(ridx, a, viol_below, viol_above, row_idcs, row_marks, nmarked);
      }
   }

   return SCIP_OKAY;
}

/** builds and adds the RLT cuts */
static
SCIP_RETCODE separateRltCuts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler */
   SCIP_SOL*             sol,                /**< the point to be separated (can be NULL) */
   SCIP_HASHMAP*         row_to_pos,         /**< hashmap linking row indices to positions in array */
   RLT_SIMPLEROW*        projrows,           /**< projected rows */
   SCIP_ROW**            rows,               /**< problem rows */
   int                   nrows,              /**< number of problem rows */
   SCIP_Bool             allowlocal,         /**< are local cuts allowed? */
   int*                  bestunderestimators,/**< indices of linear underestimators with largest violation in sol */
   int*                  bestoverestimators, /**< indices of linear overestimators with largest violation in sol */
   SCIP_RESULT*          result              /**< buffer to store whether separation was successful */
)
{
   int j;
   int r;
   int k;
   int nmarked;
   int cutssize;
   int ncuts;
   SCIP_VAR* xj;
   int* row_marks;
   int* row_idcs;
   SCIP_ROW* cut;
   SCIP_ROW** cuts;
   SCIP_Bool uselb[4] = {TRUE, TRUE, FALSE, FALSE};
   SCIP_Bool uselhs[4] = {TRUE, FALSE, TRUE, FALSE};
   SCIP_Bool success;
   SCIP_Bool infeasible;
   SCIP_Bool accepted;
   SCIP_Bool buildeqcut;
   SCIP_Bool iseqrow;

   assert(!sepadata->useprojection || projrows != NULL);
   assert(!sepadata->detecthidden || (bestunderestimators != NULL && bestoverestimators != NULL));

   ncuts = 0;
   cutssize = 0;
   cuts = NULL;
   *result = SCIP_DIDNOTFIND;

   SCIP_CALL( SCIPallocCleanBufferArray(scip, &row_marks, nrows) );
   SCIP_CALL( SCIPallocBufferArray(scip, &row_idcs, nrows) );

   /* loop through all variables that appear in bilinear products */
   for( j = 0; j < sepadata->nbilinvars && (sepadata->maxusedvars < 0 || j < sepadata->maxusedvars); ++j )
   {
      xj = sepadata->varssorted[j];

      /* mark all rows for multiplier xj */
      SCIP_CALL( markRowsXj(scip, sepadata, conshdlr, sol, j, allowlocal, row_to_pos, bestunderestimators,
                              bestoverestimators, row_marks, row_idcs, &nmarked) );

      assert(nmarked <= nrows);

      /* generate the projected cut and if it is violated, generate the actual cut */
      for( r = 0; r < nmarked; ++r )
      {
         int pos;
         int currentnunknown;
         SCIP_ROW* row;

         assert(row_marks[r] != 0);
         assert(SCIPhashmapExists(row_to_pos, (void*)(size_t) row_idcs[r])); /*lint !e571 */

         pos = SCIPhashmapGetImageInt(row_to_pos, (void*)(size_t) row_idcs[r]); /*lint !e571 */
         row = rows[pos];
         assert(SCIProwGetIndex(row) == row_idcs[r]);

         /* check whether this row and var fulfill the conditions */
         SCIP_CALL( isAcceptableRow(sepadata, row, xj, &currentnunknown, &accepted) );
         if( !accepted )
         {
            SCIPdebugMsg(scip, "rejected row %s for variable %s\n", SCIProwGetName(row), SCIPvarGetName(xj));
            row_marks[r] = 0;
            continue;
         }

         SCIPdebugMsg(scip, "accepted row %s for variable %s\n", SCIProwGetName(rows[r]), SCIPvarGetName(xj));
#ifdef SCIP_DEBUG
         SCIP_CALL( SCIPprintRow(scip, rows[r], NULL) );
#endif
         iseqrow = SCIPisEQ(scip, SCIProwGetLhs(row), SCIProwGetRhs(row));

         /* if all terms are known and it is an equality row, compute equality cuts */
         buildeqcut = (currentnunknown == 0 && iseqrow);

         /* go over all suitable combinations of sides and bounds and compute the respective cuts */
         for( k = 0; k < 4; ++k )
         {
            /* if equality cuts are possible, lhs and rhs cuts are equal so skip rhs */
            if( buildeqcut )
            {
               if( k % 2 == 1 )
                  continue;
            }
            /* otherwise which cuts are generated depends on the marks */
            else
            {
               if( row_marks[r] == 1 && uselb[k] == uselhs[k] )
                  continue;

               if( row_marks[r] == 2 && uselb[k] != uselhs[k] )
                  continue;
            }

            success = TRUE;
            cut = NULL;

            SCIPdebugMsg(scip, "row %s, uselb = %d, uselhs = %d\n", SCIProwGetName(row), uselb[k], uselhs[k]);

            /* if no variables are left in the projected row, the RLT cut will not be violated */
            if( sepadata->useprojection )
            {
               if( projrows[pos].nNonz == 0 )
                  continue;

               /* compute the rlt cut for a projected row first */
               SCIP_CALL( computeProjRltCut(scip, sepa, sepadata, &cut, projrows, pos, sol, bestunderestimators,
                     bestoverestimators, xj, &success, uselb[k], uselhs[k], allowlocal, buildeqcut) );

               /* if the projected cut is not violated, set success to FALSE */
               if( cut != NULL )
                  SCIPdebugMsg(scip, "proj cut viol = %g\n", SCIPgetRowFeasibility(scip, cut));
               if( cut != NULL && !SCIPisFeasLT(scip, SCIPgetRowFeasibility(scip, cut), 0.0) )
               {
                  SCIPdebugMsg(scip, "projected cut is not violated, feasibility = %g\n", SCIPgetRowFeasibility(scip, cut));
                  success = FALSE;
               }

               /* release the projected cut */
               if( cut != NULL )
                  SCIP_CALL( SCIPreleaseRow(scip, &cut) );
            }

            /* if we don't use projection or if the projected cut was generated successfully and is violated,
             * generate the actual cut */
            if( success )
            {
               SCIP_CALL( computeRltCuts(scip, sepa, sepadata, &cut, row, sol, bestunderestimators,
                     bestoverestimators, xj, &success, uselb[k], uselhs[k], allowlocal, buildeqcut) );
            }

            if( success )
            {
               success = SCIPisFeasLT(scip, SCIPgetRowFeasibility(scip, cut), 0.0) || (sepadata->addtopool &&
                       !SCIProwIsLocal(cut));
            }

            /* if the cut was created successfully and is violated or (if addtopool == TRUE) globally valid,
             * it is added to the cuts array */
            if( success )
            {
               if( ncuts + 1 > cutssize )
               {
                  int newsize;

                  newsize = SCIPcalcMemGrowSize(scip, ncuts + 1);
                  SCIP_CALL( SCIPreallocBufferArray(scip, &cuts, newsize) );
                  cutssize = newsize;
               }
               cuts[ncuts] = cut;
               (ncuts)++;
            }
            else
            {
               SCIPdebugMsg(scip, "the generation of the cut failed or cut not violated and not added to cutpool\n");
               /* release the cut */
               if( cut != NULL )
               {
                  SCIP_CALL( SCIPreleaseRow(scip, &cut) );
               }
            }
         }
         /* clear row_marks[r] since it will be used for the next multiplier */
         row_marks[r] = 0;
      }
   }

   /* if cuts were found, we apply an additional filtering procedure, which is similar to sepastore */
   if( ncuts > 0  )
   {
      int nselectedcuts;
      int i;

      assert(cuts != NULL);

      SCIP_CALL( SCIPselectCuts(scip, cuts, NULL, sepadata->goodscore, sepadata->badscore, sepadata->goodmaxparall,
            sepadata->maxparall, sepadata->dircutoffdistweight, sepadata->efficacyweight, sepadata->objparalweight,
            0.0, ncuts, 0, sepadata->maxncuts == -1 ? ncuts : sepadata->maxncuts, &nselectedcuts) );

      for( i = 0; i < ncuts; ++i )
      {
         assert(cuts[i] != NULL);

         if( i < nselectedcuts )
         {
            /* if selected, add global cuts to the pool and local cuts to the sepastore */
            if( SCIProwIsLocal(cuts[i]) || !sepadata->addtopool )
            {
               SCIP_CALL( SCIPaddRow(scip, cuts[i], FALSE, &infeasible) );

               if( infeasible )
               {
                  SCIPinfoMessage(scip, NULL, "CUTOFF! The cut obtained %s revealed infeasibility\n",
                        SCIProwGetName(cuts[i]));
                  *result = SCIP_CUTOFF;
               }
               else
               {
                  SCIPdebugMsg(scip, "SEPARATED: added cut to scip\n");
                  *result = SCIP_SEPARATED;
               }
            }
            else
            {
               SCIP_CALL( SCIPaddPoolCut(scip, cuts[i]) );
            }
         }

         /* release current cut */
         SCIP_CALL( SCIPreleaseRow(scip, &cuts[i]) );
      }
   }

   SCIPdebugMsg(scip, "exit separator because cut calculation is finished\n");

   SCIPfreeBufferArrayNull(scip, &cuts);
   SCIPfreeBufferArray(scip, &row_idcs);
   SCIPfreeCleanBufferArray(scip, &row_marks);

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

/** fills an array of rows suitable for RLT cut generation */
static
SCIP_RETCODE storeSuitableRows(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_ROW**            prob_rows,          /**< problem rows */
   SCIP_ROW**            rows,               /**< an array to be filled with suitable rows */
   int*                  nrows,              /**< buffer to store the number of suitable rows */
   SCIP_HASHMAP*         row_to_pos,         /**< hashmap linking row indices to positions in rows */
   SCIP_Bool             allowlocal          /**< are local rows allowed? */
   )
{
   int new_nrows;
   int r;
   int j;
   SCIP_Bool iseqrow;
   SCIP_COL** cols;
   SCIP_Bool iscontrow;

   new_nrows = 0;

   for( r = 0; r < *nrows; ++r )
   {
      iseqrow = SCIPisEQ(scip, SCIProwGetLhs(prob_rows[r]), SCIProwGetRhs(prob_rows[r]));

      /* if equality rows are requested, only those can be used */
      if( sepadata->onlyeqrows && !iseqrow )
         continue;

      /* if global cuts are requested, only globally valid rows can be used */
      if( !allowlocal && SCIProwIsLocal(prob_rows[r]) )
         continue;

      /* if continuous rows are requested, only those can be used */
      if( sepadata->onlycontrows )
      {
         cols = SCIProwGetCols(prob_rows[r]);
         iscontrow = TRUE;

         /* check row for integral variables */
         for( j = 0; j < SCIProwGetNNonz(prob_rows[r]); ++j )
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
      if( SCIProwGetOriginSepa(prob_rows[r]) == sepa || strcmp(SCIProwGetName(prob_rows[r]), "mccormick") == 0 )
         continue;

      /* if we are here, the row has passed all checks and should be added to rows */
      rows[new_nrows] = prob_rows[r];
      SCIP_CALL( SCIPhashmapSetImageInt(row_to_pos, (void*)(size_t)SCIProwGetIndex(prob_rows[r]), new_nrows) ); /*lint !e571 */
      ++new_nrows;
   }

   *nrows = new_nrows;

   return SCIP_OKAY;
}

/** adds McCormick inequalities for implicit products */
static
SCIP_RETCODE separateMcCormickImplicit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SEPADATA*        sepadata,           /**< separator data */
   SCIP_SOL*             sol,                /**< the point to be separated (can be NULL) */
   int*                  bestunderestimators,/**< indices of linear underestimators with largest violation in sol */
   int*                  bestoverestimators, /**< indices of linear overestimators with largest violation in sol */
   SCIP_RESULT*          result              /**< pointer to store the result */
   )
{
   int i;
   int j;
   SCIP_CONSEXPR_BILINTERM* terms;
   SCIP_ROW* cut;
   char name[SCIP_MAXSTRLEN];
   SCIP_Bool underestimate;
   SCIP_Real productval;
   SCIP_Real auxval;
   SCIP_Real xcoef;
   SCIP_Real ycoef;
   SCIP_Real auxcoef;
   SCIP_Real constant;
   SCIP_Bool success;
   SCIP_CONSEXPR_AUXEXPR* auxexpr;
   SCIP_Bool cutoff;
   SCIP_Real refpointx;
   SCIP_Real refpointy;
   SCIP_INTERVAL bndx;
   SCIP_INTERVAL bndy;

   assert(sepadata->nbilinterms == SCIPgetConsExprNBilinTerms(sepadata->conshdlr));
   assert(bestunderestimators != NULL && bestoverestimators != NULL);

   cutoff = FALSE;
   terms = SCIPgetConsExprBilinTerms(sepadata->conshdlr);

   for( i = 0; i < sepadata->nbilinterms; ++i )
   {
      if( terms[i].existing )
         continue;

      assert(terms[i].nauxexprs > 0);

      productval = SCIPgetSolVal(scip, sol, terms[i].x) * SCIPgetSolVal(scip, sol, terms[i].y);

      bndx.inf = SCIPvarGetLbLocal(terms[i].x);
      bndx.sup = SCIPvarGetUbLocal(terms[i].x);
      bndy.inf = SCIPvarGetLbLocal(terms[i].y);
      bndy.sup = SCIPvarGetUbLocal(terms[i].y);
      refpointx = SCIPgetSolVal(scip, sol, terms[i].x);
      refpointy = SCIPgetSolVal(scip, sol, terms[i].y);

      /* adjust the reference points */
      refpointx = MIN(MAX(refpointx, bndx.inf), bndx.sup); /*lint !e666*/
      refpointy = MIN(MAX(refpointy, bndy.inf), bndy.sup); /*lint !e666*/

      /* one iteration for underestimation and one for overestimation */
      for( j = 0; j < 2; ++j )
      {
         /* if underestimate, separate auxexpr <= x*y; if !underestimate, separate x*y <= auxexpr */
         underestimate = j == 0;
         if( underestimate && bestunderestimators[i] != -1 )
            auxexpr = terms[i].aux.exprs[bestunderestimators[i]];
         else if( !underestimate && bestoverestimators[i] != -1 )
            auxexpr = terms[i].aux.exprs[bestoverestimators[i]];
         else
            continue;

         auxval = SCIPevalConsExprBilinAuxExpr(scip, terms[i].x, terms[i].y, auxexpr, sol);

         /* skip non-violated terms */
         if( (underestimate && productval <= auxval) || (!underestimate && productval >= auxval) )
            continue;

         /* create an empty row */
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "mccormick_%sestimate_implicit_%s*%s_%d",
                             underestimate ? "under" : "over", SCIPvarGetName(terms[i].x), SCIPvarGetName(terms[i].y),
                             SCIPgetNLPs(scip));

         SCIP_CALL(SCIPcreateEmptyRowSepa(scip, &cut, sepa, name, -SCIPinfinity(scip), SCIPinfinity(scip), TRUE,
               FALSE, FALSE) );

         xcoef = 0.0;
         ycoef = 0.0;
         auxcoef = 0.0;
         constant = 0.0;
         success = TRUE;

         /* subtract auxexpr from the cut */
         addAuxexprToRow(terms[i].x, terms[i].y, auxexpr, -1.0, &auxcoef, &xcoef, &ycoef, &constant);

         /* add McCormick terms: ask for an overestimator if relation is auxexpr <= x*y, and vice versa */
         SCIPaddBilinMcCormick(scip, 1.0, bndx.inf, bndx.sup, refpointx, bndy.inf, bndy.sup, refpointy, underestimate,
               &xcoef, &ycoef, &constant, &success);

         if( REALABS(constant) > MAXVARBOUND )
            success = FALSE;

         if( success )
         {
            assert(!SCIPisInfinity(scip, REALABS(xcoef)));
            assert(!SCIPisInfinity(scip, REALABS(ycoef)));
            assert(!SCIPisInfinity(scip, REALABS(constant)));

            SCIP_CALL( SCIPaddVarToRow(scip, cut, terms[i].x, xcoef) );
            SCIP_CALL( SCIPaddVarToRow(scip, cut, terms[i].y, ycoef) );
            SCIP_CALL( SCIPaddVarToRow(scip, cut, auxexpr->auxvar, auxcoef) );

            /* set side */
            if( underestimate )
               SCIP_CALL( SCIPchgRowLhs(scip, cut, -constant) );
            else
               SCIP_CALL( SCIPchgRowRhs(scip, cut, -constant) );

            /* if the cut is violated, add it to SCIP */
            if( SCIPisFeasLT(scip, SCIPgetRowFeasibility(scip, cut), 0.0) )
            {
               SCIP_CALL( SCIPaddRow(scip, cut, FALSE, &cutoff) );
               *result = SCIP_SEPARATED;
            }
            else
            {
               SCIPdebugMsg(scip, "\nMcCormick cut for hidden product %s*%s was created successfully, but is not violated",
                            SCIPvarGetName(terms[i].x), SCIPvarGetName(terms[i].y));
            }
         }

         /* release the cut */
         if( cut != NULL )
         {
            SCIP_CALL( SCIPreleaseRow(scip, &cut) );
         }

         if( cutoff )
         {
            *result = SCIP_CUTOFF;
            SCIPdebugMsg(scip, "exit separator because we found enough cuts or a cutoff -> skip\n");
            return SCIP_OKAY;
         }
      }
   }

   return SCIP_OKAY;
}

/** LP solution separation method of separator */
static
SCIP_DECL_SEPAEXECLP(sepaExeclpRlt)
{  /*lint --e{715}*/
   SCIP_ROW** prob_rows;
   SCIP_ROW** rows;
   SCIP_SEPADATA* sepadata;
   int ncalls;
   int depth;
   int nrows;
   SCIP_HASHMAP* row_to_pos;
   RLT_SIMPLEROW* projrows;
   int* bestunderestimators;
   int* bestoverestimators;

   assert(strcmp(SCIPsepaGetName(sepa), SEPA_NAME) == 0);

   sepadata = SCIPsepaGetData(sepa);
   *result = SCIP_DIDNOTRUN;

   if( sepadata->maxncuts == 0 )
   {
      SCIPdebugMsg(scip, "exit separator because maxncuts is set to 0\n");
      return SCIP_OKAY;
   }

   /* don't run in a sub-SCIP or in probing */
   if( SCIPgetSubscipDepth(scip) > 0 && !sepadata->useinsubscip )
   {
      SCIPdebugMsg(scip, "exit separator because in sub-SCIP\n");
      return SCIP_OKAY;
   }

   /* don't run in a sub-SCIP or in probing */
   if( SCIPinProbing(scip) )
   {
      SCIPdebugMsg(scip, "exit separator because in or probing\n");
      return SCIP_OKAY;
   }

   /* only call separator a given number of times at each node */
   depth = SCIPgetDepth(scip);
   ncalls = SCIPsepaGetNCallsAtNode(sepa);
   if( (depth == 0 && sepadata->maxroundsroot >= 0 && ncalls >= sepadata->maxroundsroot)
        || (depth > 0 && sepadata->maxrounds >= 0 && ncalls >= sepadata->maxrounds) )
   {
      SCIPdebugMsg(scip, "exit separator because round limit for this node is reached\n");
      return SCIP_OKAY;
   }

   /* if this is called for the first time, create the sepadata and start the initial separation round */
   if( !sepadata->iscreated )
   {
      *result = SCIP_DIDNOTFIND;
      SCIP_CALL( createSepaData(scip, sepadata) );
   }
   assert(sepadata->iscreated || sepadata->nbilinvars == 0);
   assert(sepadata->nbilinterms == SCIPgetConsExprNBilinTerms(sepadata->conshdlr));

   /* no bilinear terms available -> skip */
   if( sepadata->nbilinvars == 0 )
   {
      SCIPdebugMsg(scip, "exit separator because there are no known bilinear terms\n");
      return SCIP_OKAY;
   }

   /* only call separator, if we are not close to terminating */
   if( SCIPisStopped(scip) )
   {
      SCIPdebugMsg(scip, "exit separator because we are too close to terminating\n");
      return SCIP_OKAY;
   }

   /* only call separator, if an optimal LP solution is at hand */
   if( SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
   {
      SCIPdebugMsg(scip, "exit separator because there is no LP solution at hand\n");
      return SCIP_OKAY;
   }

   /* get the rows, depending on settings */
   if( sepadata->isinitialround || sepadata->onlyinitial )
   {
      SCIP_CALL( getInitialRows(scip, &prob_rows, &nrows) );
   }
   else
   {
      SCIP_CALL( SCIPgetLPRowsData(scip, &prob_rows, &nrows) );
   }

   /* save the suitable rows */
   SCIP_CALL( SCIPallocBufferArray(scip, &rows, nrows) );
   SCIP_CALL( SCIPhashmapCreate(&row_to_pos, SCIPblkmem(scip), nrows) );

   SCIP_CALL( storeSuitableRows(scip, sepa, sepadata, prob_rows, rows, &nrows, row_to_pos, allowlocal) );

   if( sepadata->isinitialround || sepadata->onlyinitial )
   {
      SCIPfreeBufferArray(scip, &prob_rows);
      sepadata->isinitialround = FALSE;
   } /* no need to free memory in the other case since SCIPgetLPRowsData does not allocate it */

   if( nrows == 0 ) /* no suitable rows found, free memory and exit */
   {
      SCIPhashmapFree(&row_to_pos);
      SCIPfreeBufferArray(scip, &rows);
      return SCIP_OKAY;
   }
   else /* suitable rows have been found */
   {
      SCIP_CALL( SCIPreallocBufferArray(scip, &rows, nrows) );
   }

   /* create the projected problem */
   if( sepadata->useprojection )
   {
      SCIP_Bool allcst;

      SCIP_CALL( createProjRows(scip, rows, nrows, NULL, &projrows, allowlocal, &allcst) );

      /* if all projected rows have only constants left, quit */
      if( allcst )
         goto TERMINATE;

#ifdef SCIP_DEBUG
      printProjRows(scip, projrows, nrows, NULL);
#endif
   }
   else
   {
      projrows = NULL;
   }

   /* separate the cuts */
   if( sepadata->detecthidden )
   {
      /* if we detect implicit products, a term might have more than one estimator in each direction;
       * save the indices of the most violated estimators
       */
      SCIP_CALL( SCIPallocBufferArray(scip, &bestunderestimators, sepadata->nbilinterms) );
      SCIP_CALL( SCIPallocBufferArray(scip, &bestoverestimators, sepadata->nbilinterms) );
      getBestEstimators(scip, sepadata, NULL, bestunderestimators, bestoverestimators);

      /* also separate McCormick cuts for implicit products */
      SCIP_CALL( separateMcCormickImplicit(scip, sepa, sepadata, NULL, bestunderestimators, bestoverestimators,
            result) );

      if( *result != SCIP_CUTOFF )
      {
         SCIP_CALL( separateRltCuts(scip, sepa, sepadata, sepadata->conshdlr, NULL, row_to_pos, projrows, rows, nrows,
               allowlocal, bestunderestimators, bestoverestimators, result) );
      }
   }
   else
   {
      SCIP_CALL( separateRltCuts(scip, sepa, sepadata, sepadata->conshdlr, NULL, row_to_pos, projrows, rows, nrows,
            allowlocal, NULL, NULL, result) );
   }

   if( sepadata->detecthidden )
   {
      SCIPfreeBufferArray(scip, &bestoverestimators);
      SCIPfreeBufferArray(scip, &bestunderestimators);
   }

 TERMINATE:
   /* free the projected problem */
   if( sepadata->useprojection )
   {
      freeProjRows(scip, &projrows, nrows);
   }

   SCIPhashmapFree(&row_to_pos);
   SCIPfreeBufferArray(scip, &rows);

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

   SCIP_CALL( SCIPaddIntParam(scip,
      "separating/" SEPA_NAME "/maxrounds",
      "maximal number of separation rounds per node (-1: unlimited)",
      &sepadata->maxrounds, FALSE, DEFAULT_MAXROUNDS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
      "separating/" SEPA_NAME "/maxroundsroot",
      "maximal number of separation rounds in the root node (-1: unlimited)",
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

   SCIP_CALL( SCIPaddBoolParam(scip,
                               "separating/" SEPA_NAME "/useprojection",
      "if set to true, projected rows are checked first",
      &sepadata->useprojection, FALSE, DEFAULT_USEPROJECTION, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
                               "separating/" SEPA_NAME "/detecthidden",
      "if set to true, hidden products are detected and separated by McCormick cuts",
      &sepadata->detecthidden, FALSE, DEFAULT_DETECTHIDDEN, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
                               "separating/" SEPA_NAME "/hiddenrlt",
      "whether RLT cuts (TRUE) or only McCormick inequalities (FALSE) should be added for hidden products",
      &sepadata->hiddenrlt, FALSE, DEFAULT_HIDDENRLT, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
                               "separating/" SEPA_NAME "/addtopool",
      "if set to true, globally valid RLT cuts are added to the global cut pool",
      &sepadata->addtopool, FALSE, DEFAULT_ADDTOPOOL, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip,
                               "separating/" SEPA_NAME "/goodscore",
      "threshold for score of cut relative to best score to be considered good, so that less strict filtering is applied",
      &sepadata->goodscore, TRUE, DEFAULT_GOODSCORE, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
                               "separating/" SEPA_NAME "/badscore",
      "threshold for score of cut relative to best score to be discarded",
      &sepadata->badscore, TRUE, DEFAULT_BADSCORE, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
                               "separating/" SEPA_NAME "/objparalweight",
      "weight of objective parallelism in cut score calculation",
      &sepadata->objparalweight, TRUE, DEFAULT_OBJPARALWEIGHT, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
                               "separating/" SEPA_NAME "/efficacyweight",
      "weight of efficacy in cut score calculation",
      &sepadata->efficacyweight, TRUE, DEFAULT_EFFICACYWEIGHT, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
                               "separating/" SEPA_NAME "/dircutoffdistweight",
      "weight of directed cutoff distance in cut score calculation",
      &sepadata->dircutoffdistweight, TRUE, DEFAULT_DIRCUTOFFDISTWEIGHT, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
                               "separating/" SEPA_NAME "/goodmaxparall",
      "maximum parallelism for good cuts",
      &sepadata->goodmaxparall, TRUE, DEFAULT_GOODMAXPARALL, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
                               "separating/" SEPA_NAME "/maxparall",
      "maximum parallelism for non-good cuts",
      &sepadata->maxparall, TRUE, DEFAULT_MAXPARALL, 0.0, 1.0, NULL, NULL) );

   return SCIP_OKAY;
}
