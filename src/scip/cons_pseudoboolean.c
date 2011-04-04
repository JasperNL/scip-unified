/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2011 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_pseudoboolean.c
 * @ingroup CONSHDLRS 
 * @brief  constraint handler for pseudo Boolean constraints
 * @author Stefan Heinz
 * @author Michael Winkler
 *
 *
 * The constraint handler deals with pseudo Boolean constraints. These are constraints of the form 
 *
 * lhs <= \sum_{k=0}^m c_k * x_k  +  \sum_{i=0}^n c_i * \prod_{j\in I_i} x_j <= rhs
 *
 * where all x are binary and all c are integer
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/cons_pseudoboolean.h"
#include "scip/cons_and.h"
#include "scip/cons_indicator.h"
#if 0
#include "scip/cons_eqknapsack.h"
#endif
#include "scip/cons_knapsack.h"
#include "scip/cons_linear.h"
#include "scip/cons_logicor.h"
#include "scip/cons_setppc.h"
#include "scip/pub_var.h"


/* constraint handler properties */
#define CONSHDLR_NAME          "pseudoboolean"
#define CONSHDLR_DESC          "constraint handler template"
#define CONSHDLR_SEPAPRIORITY  +1000000 /**< priority of the constraint handler for separation */
#define CONSHDLR_ENFOPRIORITY  -1000000 /**< priority of the constraint handler for constraint enforcing */
#define CONSHDLR_CHECKPRIORITY -5000000 /**< priority of the constraint handler for checking feasibility */
#define CONSHDLR_SEPAFREQ            -1 /**< frequency for separating cuts; zero means to separate only in the root node */
#define CONSHDLR_PROPFREQ            -1 /**< frequency for propagating domains; zero means only preprocessing propagation */
#define CONSHDLR_EAGERFREQ          100 /**< frequency for using all instead of only the useful constraints in separation,
                                              *   propagation and enforcement, -1 for no eager evaluations, 0 for first only */
#define CONSHDLR_MAXPREROUNDS        -1 /**< maximal number of presolving rounds the constraint handler participates in (-1: no limit) */
#define CONSHDLR_DELAYSEPA        FALSE /**< should separation method be delayed, if other separators found cuts? */
#define CONSHDLR_DELAYPROP        FALSE /**< should propagation method be delayed, if other propagators found reductions? */
#define CONSHDLR_DELAYPRESOL      FALSE /**< should presolving method be delayed, if other presolvers found reductions? */
#define CONSHDLR_NEEDSCONS         TRUE /**< should the constraint handler be skipped, if no constraints are available? */

#define DEFAULT_DECOMPOSENORMALPBCONS FALSE /**< decompose all normal pseudo boolean constraint into a "linear" constraint "and" constrainst */
#define DEFAULT_DECOMPOSEINDICATORPBCONS TRUE /**< decompose all indicator pseudo boolean constraint into a "linear" constraint "and" constrainst */
#define DEFAULT_SEPARATENONLINEAR  TRUE /**< if decomposed, should the nonlinear constraints be separated during LP processing */
#define DEFAULT_PROPAGATENONLINEAR TRUE /**< if decomposed, should the nonlinear constraints be propagated during node processing */
#define DEFAULT_REMOVABLENONLINEAR TRUE /**< if decomposed, should the nonlinear constraints be removable */
#define USEINDICATOR               TRUE

/*
 * Data structures
 */
#define HASHSIZE_PSEUDOBOOLEANNONLINEARTERMS 131101 /**< minimal size of hash table in and constraint tables */


/* TODO: - create special linear(knapsack, setppc, logicor, (eqknapsack)) and and-constraints with check flags FALSE, to
 *         get smaller amount of locks on the term variables, do all presolving ...?! in these constraint handlers
 *
 *       - do the checking here, lock and-resultants in both directions and all and-variables according to their
 *         coefficients and sides of the constraint, @Note: this only works if the and-resultant has no objective
 *         cofficient, otherwise we need to lock variables also in both directions
 *
 *       - need to keep and constraint pointer for special propagations like if two ands are due to their variables in
 *         one clique, add this cliques of and-resultants
 *
 *       - do special presolving like on instance : 
 * check/IP/PseudoBoolean/normalized-PB07/OPT-SMALLINT-NLC/submittedPB07/manquinho/bsg/normalized-bsg_1000_25_1.opb.gz
 *
 *         there exist constraint like:        1 x1 x2 + 1 x1 x3 + 1 x1 x4 + 1 x1 x5 <= 1 ;
 *         which "equals" a linear constraint: 3 x1 + x2 + x3 + x4 + x5 <= 4 ;
 *
 *         in more general terms:                     1 x1 x2 x3 x4 + 1 x1 x2 x5 x6 x7 + 1 x1 x2 x8 x9 <= 1 ;
 *         which "equals" a pseudoboolean constraint: 2 x1 + 2 x2 + 1 x3 x4 + 1 x5 x6 x7 + 1 x8 x9 <= 5 ;
 *
 *         in an even more general terms:             5 x1 x2 x3 x4 + 1 x1 x2 x5 x6 x7 + 1 x1 x2 x8 x9 <= 6 ;
 *                   equals(should the knapsack do)   1 x1 x2 x3 x4 + 1 x1 x2 x5 x6 x7 + 1 x1 x2 x8 x9 <= 2 ;
 *         which "equals" a pseudoboolean constraint: 2 x1 + 2 x2 + 1 x3 x4 + 1 x5 x6 x7 + 1 x8 x9 <= 6 ;
 *         (         without knapsack                 7 x1 + 7 x2 + 5 x3 x4 + 1 x5 x6 x7 + 1 x8 x9 <= 20 ; )
 *
 *         another special case :                     1 x1 x2 x3 + 1 x1 x2 x4 + 1 x5 x6 <= 1 ;
 *         which "equals" a pseudoboolean constraint: 2 x1 + 2 x2 + 1 x3 + 1 x4 + 1 x5 x6 <= 5 ;
 *         which "equals" a pseudoboolean constraint: 4 x1 + 4 x2 + 2 x3 + 2 x4 + 1 x5 + 1 x6 <= 10 ;
 *
 *         another special case :                     1 x1 x2 + 1 x1 x3 + 2 x4 x5 <= 3 ;
 *         which "equals" a pseudoboolean constraint: 2 x1 + 1 x2 + 1 x3 + 2 x4 x5 <= 5 ;
 *         which "equals" a pseudoboolean constraint: 2 x1 + 1 x2 + 1 x3 + 1 x4 + 1 x5 <= 5 ;
 *
 *       - move mergeMultiplesNonLinear and check for cliques inside an and-constraint handler 
 *
 *       - in and-constraint better count nfixed zeros in both directions and maybe nfixedones for better propagation
 *
 *       - do better conflict analysis by choosing the earliest fixed variable which led to a conflict instead of maybe
 *         best coefficient or create more conflicts by using all to zero fixed variables one by one
 *
 *       - how to make sure that we aggregate in a right way, when aggregating a resultant and a "normal" variable, 
 *         maybe add in SCIPaggregateVars a check for original variables, to prefer them if the variable type is the
 *         same; probably it would be better too if we would aggregate two resultants that the one with less variables
 *         inside the and-constraint will stay active
 */


/** and-constraint data object */
struct ConsAndData
{
   SCIP_CONS*           cons;                /**< pointer to the and-constraint of this 'term' of variables */
   SCIP_CONS*           origcons;            /**< pointer to the original and-constraint of this 'term' of variables
                                              *   after problem was transformed, NULL otherwise */
   SCIP_VAR**           vars;                /**< all variables */
   int                  nvars;               /**< number of all variables */
   int                  svars;               /**< size of all variables */
   SCIP_VAR**           newvars;             /**< new variables in this presolving round */
   int                  nnewvars;            /**< number of new variables in this presolving round */
   int                  snewvars;            /**< size of new variables in this presolving round */
   int                  nuses;               /**< how often is this data in usage */
   SCIP_Bool            deleted;             /**< was memory of both variable arrays already freed */
};
typedef struct ConsAndData CONSANDDATA;

/** constraint data for pseudoboolean constraints */
struct SCIP_ConsData
{
   SCIP_Real             lhs;                /**< left hand side of constraint */
   SCIP_Real             rhs;                /**< right hand side of constraint */
   
   SCIP_CONS*            lincons;            /**< linear constraint which represents this pseudoboolean constraint */
   SCIP_LINEARCONSTYPE   linconstype;        /**< type of linear constraint which represents this pseudoboolean constraint */
   int                   nlinvars;           /**< number of linear variables (without and-resultants) */

   CONSANDDATA**         consanddatas;       /**< array of and-constraints-data-objects sorted after and-resultant of
                                              *   corresponding and-constraint */
   SCIP_Real*            andcoefs;           /**< array of coefficients for and-constraints of
                                              *   and-constraints-data-objects before sorted the same way like above
                                              *   (changes in this presolving round, need to update in every presolving
                                              *   round) */
   int                   nconsanddatas;      /**< number of and-constraints-data-objects */
   int                   sconsanddatas;      /**< size of and-constraints-data-objects array */

   SCIP_VAR*             intvar;             /**< a artificial variable which was added only for the objective function,
                                              *   if this variable is not NULL this constraint (without this integer
                                              *   variable) describes the objective funktion */

   SCIP_VAR*             indvar;             /**< indicator variable if it's a soft constraint, or NULL */
   SCIP_Real             weight;             /**< weight of the soft constraint, if it is one */

   unsigned int          issoftcons:1;       /**< is this a soft constraint */
   unsigned int          changed:1;          /**< was constraint changed? */
   unsigned int          propagated:1;       /**< is constraint already propagated? */
   unsigned int          presolved:1;        /**< is constraint already presolved? */
   unsigned int          cliquesadded:1;     /**< were the cliques of the constraint already extracted? */
   unsigned int          upgradetried:1;     /**< was constraint upgrading already tried */
};

/** constraint handler data */
struct SCIP_ConshdlrData
{
   CONSANDDATA**         allconsanddatas;    /**< array of all and-constraint data objects inside the whole problem,
                                              *   created via this constraint handler */ 
   int                   nallconsanddatas;   /**< number of all and-constraint data objects inside the whole problem,
                                              *   created via this constraint handler */
   int                   sallconsanddatas;   /**< size of all and-constraint data objects inside the whole problem,
                                              *   created via this constraint handler */
   SCIP_HASHTABLE*       hashtable;          /**< hash table for all and-constraint data objects */
   int                   hashtablesize;      /**< size for hash table for all and-constraint data objects */

   SCIP_HASHMAP*         hashmap;            /**< hash map for mapping all resultant to and-constraint */
   int                   hashmapsize;        /**< size for hash map for mapping all resultant to and-constraint */

   SCIP_Bool             decomposenormalpbcons;/**< decompose the pseudo boolean constraint into a "linear" constraint "and" constrainst */
   SCIP_Bool             decomposeindicatorpbcons;/**< decompose the indicator pseudo boolean constraint into a "linear" constraint "and" constrainst */
   int                   nlinconss;          /**< for counting number of created linear constraints */
};

/*
 * Local methods
 */


/** gets the key of the given element */
static
SCIP_DECL_HASHGETKEY(hashGetKeyAndConsDatas)
{  /*lint --e{715}*/
   /* the key is the element itself */ 
   return elem;
}

/** returns TRUE iff both keys are equal; two non-linear terms are equal if they have the same variables */
static
SCIP_DECL_HASHKEYEQ(hashKeyEqAndConsDatas)
{
   SCIP* scip;
   CONSANDDATA* cdata1;
   CONSANDDATA* cdata2;
   int v;

   cdata1 = (CONSANDDATA*)key1;
   cdata2 = (CONSANDDATA*)key2;
   scip = (SCIP*)userptr; 

   assert(scip != NULL);
   assert(cdata1 != NULL);
   assert(cdata2 != NULL);
   assert(cdata1->vars != NULL);
   assert(cdata1->nvars > 1);
   assert(cdata2->vars != NULL);
   assert(cdata2->nvars > 1);

#ifndef NDEBUG
   /* check that cdata1 variables are sorted */
   for( v = cdata1->nvars - 1; v > 0; --v )
      assert(SCIPvarGetIndex(cdata1->vars[v]) >= SCIPvarGetIndex(cdata1->vars[v - 1]));
   /* check that cdata2 variables are sorted */
   for( v = cdata2->nvars - 1; v > 0; --v )
      assert(SCIPvarGetIndex(cdata2->vars[v]) >= SCIPvarGetIndex(cdata2->vars[v - 1]));
#endif

   /* checks trivial case */
   if( cdata1->nvars != cdata2->nvars )
      return FALSE;

   /* checks trivial case */
   if( cdata1->cons != NULL && cdata2->cons != NULL && cdata1->cons != cdata2->cons )
      return FALSE;

   /* check each variable in both cdatas for equality */
   for( v = cdata1->nvars - 1; v >= 0; --v )
   {
      assert(cdata1->vars[v] != NULL);
      assert(cdata2->vars[v] != NULL);

      /* tests if variables are equal */
      if( cdata1->vars[v] != cdata2->vars[v] )
      {
         assert(SCIPvarCompare(cdata1->vars[v], cdata2->vars[v]) == 1 || 
            SCIPvarCompare(cdata1->vars[v], cdata2->vars[v]) == -1);
         return FALSE;
      }
      assert(SCIPvarCompare(cdata1->vars[v], cdata2->vars[v]) == 0); 
   } 
   
   return TRUE;
}

/** returns the hash value of the key */
static
SCIP_DECL_HASHKEYVAL(hashKeyValAndConsDatas)
{  /*lint --e{715}*/
   CONSANDDATA* cdata;
   int minidx;
   int mididx;
   int maxidx;
   
   cdata = (CONSANDDATA*)key;
   
   assert(cdata != NULL);
   assert(cdata->vars != NULL);
   assert(cdata->nvars > 1);
#ifndef NDEBUG
   {
      /* check that these variables are sorted */
      int v;
      for( v = cdata->nvars - 1; v > 0; --v )
         assert(SCIPvarGetIndex(cdata->vars[v]) >= SCIPvarGetIndex(cdata->vars[v - 1]));
   }
#endif

   minidx = SCIPvarGetIndex(cdata->vars[0]);
   mididx = SCIPvarGetIndex(cdata->vars[cdata->nvars / 2]);
   maxidx = SCIPvarGetIndex(cdata->vars[cdata->nvars - 1]);
   assert(minidx >= 0 && minidx <= maxidx);

   return (cdata->nvars << 29) + (minidx << 22) + (mididx << 11) + maxidx; /*lint !e701*/
}

/** creates constaint handler data for pseudo boolean constraint handler */
static
SCIP_RETCODE conshdlrdataCreate(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLRDATA**   conshdlrdata        /**< pointer to store the constraint handler data */
   )
{
   assert(scip != NULL);
   assert(conshdlrdata != NULL);

   SCIP_CALL( SCIPallocMemory(scip, conshdlrdata) );

   (*conshdlrdata)->allconsanddatas = NULL;
   (*conshdlrdata)->nallconsanddatas = 0;
   (*conshdlrdata)->sallconsanddatas = 10;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &((*conshdlrdata)->allconsanddatas), (*conshdlrdata)->sallconsanddatas ) );

   /* create a hash table for and-constraint data objects */
   (*conshdlrdata)->hashtablesize = SCIPcalcHashtableSize(HASHSIZE_PSEUDOBOOLEANNONLINEARTERMS);
   SCIP_CALL( SCIPhashtableCreate(&((*conshdlrdata)->hashtable), SCIPblkmem(scip), (*conshdlrdata)->hashtablesize,
         hashGetKeyAndConsDatas, hashKeyEqAndConsDatas, hashKeyValAndConsDatas, (void*) scip) );

   /* create a hash table for and-resultant to and-constraint data objects */
   (*conshdlrdata)->hashmapsize = SCIPcalcHashtableSize(HASHSIZE_PSEUDOBOOLEANNONLINEARTERMS);
   SCIP_CALL( SCIPhashmapCreate(&((*conshdlrdata)->hashmap), SCIPblkmem(scip), (*conshdlrdata)->hashmapsize) );

   /* for constraint names count number of created constraints */
   (*conshdlrdata)->nlinconss = 0;

   return SCIP_OKAY;
}


/** frees constraint handler data for pseudo boolean constraint handler */
static
SCIP_RETCODE conshdlrdataFree(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLRDATA**   conshdlrdata        /**< pointer to the constraint handler data */
   )
{
   CONSANDDATA** allconsanddatas;
   int c;
   int nallconsanddatas;
   int sallconsanddatas;

   assert(scip != NULL);
   assert(conshdlrdata != NULL);
   assert(*conshdlrdata != NULL);

   allconsanddatas = (*conshdlrdata)->allconsanddatas;
   nallconsanddatas = (*conshdlrdata)->nallconsanddatas;
   sallconsanddatas = (*conshdlrdata)->sallconsanddatas;

   for( c = nallconsanddatas - 1; c >= 0; --c )
   {
      /* free variables arrays */
      SCIPfreeBlockMemoryArrayNull(scip, &(allconsanddatas[c]->vars), allconsanddatas[c]->svars);
      allconsanddatas[c]->nvars = 0;
      allconsanddatas[c]->svars = 0;

      if( allconsanddatas[c]->snewvars > 0 )
      {
         SCIPfreeBlockMemoryArrayNull(scip, &(allconsanddatas[c]->newvars), allconsanddatas[c]->snewvars);
         allconsanddatas[c]->nnewvars = 0;
         allconsanddatas[c]->snewvars = 0;
      }
   }

   /* free hash table */
   SCIPhashmapFree(&((*conshdlrdata)->hashmap));
   (*conshdlrdata)->hashmapsize = 0;
   SCIPhashtableFree(&((*conshdlrdata)->hashtable));
   (*conshdlrdata)->hashtablesize = 0;

   SCIPfreeBlockMemoryArray(scip, &allconsanddatas, sallconsanddatas );

   (*conshdlrdata)->allconsanddatas = NULL;
   (*conshdlrdata)->nallconsanddatas = 0;
   (*conshdlrdata)->sallconsanddatas = 0;

   SCIPfreeMemory(scip, conshdlrdata);

   return SCIP_OKAY;
}

/* clears constraint handler data */
static
SCIP_RETCODE conshdlrdataClear(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLRDATA**   conshdlrdata        /**< pointer to the constraint handler data */
   )
{
   CONSANDDATA** allconsanddatas;
   int c;
   int nallconsanddatas;
   int sallconsanddatas;

   assert(scip != NULL);
   assert(conshdlrdata != NULL);
   assert(*conshdlrdata != NULL);

   allconsanddatas = (*conshdlrdata)->allconsanddatas;
   nallconsanddatas = (*conshdlrdata)->nallconsanddatas;
   sallconsanddatas = (*conshdlrdata)->sallconsanddatas;

   for( c = nallconsanddatas - 1; c >= 0; --c )
   {
      /* free variables arrays */
      SCIPfreeBlockMemoryArrayNull(scip, &(allconsanddatas[c]->vars), allconsanddatas[c]->svars);
      allconsanddatas[c]->nvars = 0;
      allconsanddatas[c]->svars = 0;

      if( allconsanddatas[c]->snewvars > 0 )
      {
         SCIPfreeBlockMemoryArrayNull(scip, &(allconsanddatas[c]->newvars), allconsanddatas[c]->snewvars);
         allconsanddatas[c]->nnewvars = 0;
         allconsanddatas[c]->snewvars = 0;
      }
   }

   /* clear hash map */
   SCIPhashmapRemoveAll((*conshdlrdata)->hashmap);

   /* clear hash table */
   SCIPhashtableRemoveAll((*conshdlrdata)->hashtable);

   /* reset number of consanddata elements */
   (*conshdlrdata)->nallconsanddatas = 0;

   return SCIP_OKAY;
}

/** gets number of variables in linear constraint */
static
SCIP_RETCODE getLinearConsNVars(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< linear constraint */
   SCIP_LINEARCONSTYPE const constype,       /**< linear constraint type */
   int*const             nvars               /**< pointer to store number variables of linear constraint */
   )
{
   assert(scip != NULL);
   assert(cons != NULL);
   assert(nvars != NULL);

   /* determine for each special linear constrait all variables and coefficients */
   switch( constype )
   {
   case SCIP_LINEAR:
      *nvars = SCIPgetNVarsLinear(scip, cons);
      break;
   case SCIP_LOGICOR:
      *nvars = SCIPgetNVarsLogicor(scip, cons);
      break;
   case SCIP_KNAPSACK:
      *nvars = SCIPgetNVarsKnapsack(scip, cons);
      break;
   case SCIP_SETPPC:
      *nvars = SCIPgetNVarsSetppc(scip, cons);
      break;
#if 0
   case SCIP_EQKNAPSACK:
      *nvars = SCIPgetNVarsEQKnapsack(scip, cons);
      break;
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** creates a pseudo boolean constraint data */
static
SCIP_RETCODE consdataCreate(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*const   conshdlr,           /**< pseudoboolean constraint handler */
   SCIP_CONSDATA**       consdata,           /**< pointer to linear constraint data */
   SCIP_CONS*const       lincons,            /**< linear constraint with artificial and-resultants representing this pseudoboolean constraint */
   SCIP_LINEARCONSTYPE const linconstype,    /**< type of linear constraint */
   SCIP_CONS**const      andconss,           /**< array of and-constraints which occur in this pseudoboolean constraint */
   SCIP_Real*const       andcoefs,           /**< coefficients of and-constraints */
   int const             nandconss,          /**< number of and-constraints */
   SCIP_VAR*const        indvar,             /**< indicator variable if it's a soft constraint, or NULL */
   SCIP_Real const       weight,             /**< weight of the soft constraint, if it is one */
   SCIP_Bool const       issoftcons,         /**< is this a soft constraint */
   SCIP_VAR* const       intvar,             /**< a artificial variable which was added only for the objective function,
                                              *   if this variable is not NULL this constraint (without this integer
                                              *   variable) describes the objective funktion */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs                 /**< right hand side of row */
   )
{
   int nvars;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(consdata != NULL);
   assert(lincons != NULL && linconstype > SCIP_INVALIDCONS);
   assert(nandconss == 0 || (andconss != NULL && andcoefs != NULL));
   assert(!issoftcons || (!SCIPisZero(scip, weight) && indvar != NULL));

   /* adjust right hand side */
   if( SCIPisInfinity(scip, rhs) )
      rhs = SCIPinfinity(scip);
   else if( SCIPisInfinity(scip, -rhs) )
      rhs = -SCIPinfinity(scip);
   
   /* adjust left hand side */
   if( SCIPisInfinity(scip, -lhs) )
      lhs = -SCIPinfinity(scip);
   else if( SCIPisInfinity(scip, lhs) )
      lhs = SCIPinfinity(scip);

   /* check left and right side */
   if( SCIPisGT(scip, lhs, rhs) )
   {
      SCIPerrorMessage("left hand side of pseudo boolean constraint greater than right hand side\n");
      SCIPerrorMessage(" -> lhs=%g, rhs=%g\n", lhs, rhs);
      return SCIP_INVALIDDATA;
   }

   /* allocate memory for the constraint data */
   SCIP_CALL( SCIPallocBlockMemory(scip, consdata) );

   /* initialize the weights for soft constraints */
   (*consdata)->issoftcons = issoftcons;
   if( issoftcons )
   {
      (*consdata)->weight = weight;
      if( SCIPisTransformed(scip) )
      {
         SCIP_CALL( SCIPgetTransformedVar(scip, indvar, &((*consdata)->indvar)) );
      }
      else
         (*consdata)->indvar = indvar;
   }
   else
      (*consdata)->indvar = NULL;
  
   /* copy artificial integer variable if it exist */
   if( intvar != NULL )
   { 
      if( SCIPisTransformed(scip) )
      {
         SCIP_CALL( SCIPgetTransformedVar(scip, intvar, &((*consdata)->intvar)) );
      }
      else
         (*consdata)->intvar = intvar;
   }
   else
      (*consdata)->intvar = NULL;
   
   /* copy linear constraint */
   (*consdata)->lincons = lincons;
   (*consdata)->linconstype = linconstype;

   /* get number of non-linear terms in pseudoboolean constraint */
   SCIP_CALL( getLinearConsNVars(scip, (*consdata)->lincons, (*consdata)->linconstype, &nvars) );
   (*consdata)->nlinvars = nvars - nandconss;

   /* copy and-constraints */
   if( nandconss > 0 )
   { 
      SCIP_CONSHDLRDATA* conshdlrdata;
      SCIP_VAR** andress;
      int c;

      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &((*consdata)->consanddatas), nandconss) );
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &((*consdata)->andcoefs), andcoefs, nandconss) );
      (*consdata)->nconsanddatas = nandconss;
      (*consdata)->sconsanddatas = nandconss;

      /* allocate temporary memory */
      SCIP_CALL( SCIPallocBufferArray(scip, &andress, nandconss) );

      conshdlrdata = SCIPconshdlrGetData(conshdlr);
      assert(conshdlrdata != NULL);
      assert(conshdlrdata->hashmap != NULL);

      /* get all and-resultants for sorting */
      for( c = nandconss - 1; c >= 0; --c )
      {
         andress[c] = SCIPgetResultantAnd(scip, andconss[c]);
         assert(andress[c] != NULL);

         (*consdata)->consanddatas[c] = (CONSANDDATA*) SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)andress[c]);
         assert((*consdata)->consanddatas[c] != NULL);
      }

      /* sort and-constraints after indices of corresponding and-resultants */
      SCIPsortPtrPtrReal((void**)andress, (void**)((*consdata)->consanddatas), (*consdata)->andcoefs, SCIPvarComp, nandconss);

      /* free temporary memory */
      SCIPfreeBufferArray(scip, &andress);
   }
   else
   {
      (*consdata)->consanddatas = NULL;
      (*consdata)->andcoefs = NULL;
      (*consdata)->nconsanddatas = 0;
      (*consdata)->sconsanddatas = 0;
   }

   /* copy left and right hand side */
   (*consdata)->lhs = lhs;
   (*consdata)->rhs = rhs;

   (*consdata)->changed = TRUE;
   (*consdata)->propagated = FALSE;
   (*consdata)->presolved = FALSE;
   (*consdata)->cliquesadded = FALSE;
   (*consdata)->upgradetried = TRUE;

   return SCIP_OKAY;
}

/** free a pseudo boolean constraint data */
static
SCIP_RETCODE consdataFree(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSDATA**       consdata            /**< pointer to linear constraint data */
   )
{
#if 0
   int c;
#endif

   assert(scip != NULL);
   assert(consdata != NULL);
   assert(*consdata != NULL);

   assert((*consdata)->nconsanddatas == 0 || (*consdata)->consanddatas != NULL);

#if 0
   for( c = (*consdata)->nconsanddatas - 1; c >= 0; --c )
   {
      CONSANDDATA* consanddata;

      consanddata = (*consdata)->consanddatas[c];
      assert(consanddata != NULL);
      
      if( consanddata->deleted )
         continue;

      assert(consanddata->nuses >= 0);
      if( consanddata->nuses > 0 )
         --(consanddata->nuses);

      /* if data object is not used anymore, delete it */
      if( consanddata->nuses == 0 )
      {
         SCIP_VAR** tmpvars;
         int v;

         tmpvars = consanddata->vars;

         /* release all old variables */
         for( v = consanddata->nvars - 1; v >= 0; --v )
         {
            assert(tmpvars[v] != NULL);
            SCIP_CALL( SCIPreleaseVar(scip, &tmpvars[v]) );
         }

         tmpvars = consanddata->newvars;

         /* release all new variables */
         for( v = consanddata->nnewvars - 1; v >= 0; --v )
         {
            assert(tmpvars[v] != NULL);
            SCIP_CALL( SCIPreleaseVar(scip, &tmpvars[v]) );
         }

         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->vars), consanddata->svars);
         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->newvars), consanddata->snewvars);

         /* delete and release and-constraint */
         SCIP_CALL( SCIPdelCons(scip, consanddata->cons) );
         SCIP_CALL( SCIPreleaseCons(scip, &(consanddata->cons)) );

         consanddata->nvars = 0;
         consanddata->svars = 0;
         consanddata->nnewvars = 0;
         consanddata->snewvars = 0;
         consanddata->deleted = TRUE;
      }
   }
#endif
   /* release linear constraint */
   if( (*consdata)->lincons != NULL )
   {
      SCIP_CALL( SCIPreleaseCons(scip, &((*consdata)->lincons)) );
   }

   /* free array of and-constraints */
   SCIPfreeBlockMemoryArrayNull(scip, &((*consdata)->andcoefs), (*consdata)->sconsanddatas);
   SCIPfreeBlockMemoryArrayNull(scip, &((*consdata)->consanddatas), (*consdata)->sconsanddatas);

   SCIPfreeBlockMemory(scip, consdata);

   return SCIP_OKAY;
}

/** gets sides of linear constraint */
static
SCIP_RETCODE getLinearConsSides(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< linear constraint */
   SCIP_LINEARCONSTYPE const constype,       /**< linear constraint type */
   SCIP_Real*const       lhs,                /**< pointer to store left hand side of linear constraint */
   SCIP_Real*const       rhs                 /**< pointer to store right hand side of linear constraint */
   )
{
   SCIP_SETPPCTYPE type;

   switch( constype )
   {
   case SCIP_LINEAR:
      *lhs = SCIPgetLhsLinear(scip, cons);
      *rhs = SCIPgetRhsLinear(scip, cons);
      break;
   case SCIP_LOGICOR:
      *lhs = 1.0;
      *rhs = SCIPinfinity(scip);
      break;
   case SCIP_KNAPSACK:
      *lhs = -SCIPinfinity(scip);
      *rhs = SCIPgetCapacityKnapsack(scip, cons);
      break;
   case SCIP_SETPPC:
      type = SCIPgetTypeSetppc(scip, cons);
      
      switch( type )
      {
      case SCIP_SETPPCTYPE_PARTITIONING:
         *lhs = 1.0;
         *rhs = 1.0;
         break;
      case SCIP_SETPPCTYPE_PACKING:
         *lhs = -SCIPinfinity(scip);
         *rhs = 1.0;
         break;
      case SCIP_SETPPCTYPE_COVERING:
         *lhs = 1.0;
         *rhs = SCIPinfinity(scip);
         break;
      default:
         SCIPerrorMessage("unknown setppc type\n");
         return SCIP_INVALIDDATA;
      }
      break;
#if 0
   case SCIP_EQKNAPSACK:
      *lhs = SCIPgetCapacityEQKnapsack(scip, cons);
      *rhs = *lhs;
      break;
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** installs rounding locks for the given and-constraint associated with given coefficient */
static
SCIP_RETCODE lockRoundingAndCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   CONSANDDATA*const     consanddata,        /**< CONSANDDATA object for which we want to delete the locks and the
                                              *   capture of the corresponding and-constraint */
   SCIP_Real const       coef,               /**< coefficient which led to old locks */
   SCIP_Real const       lhs,                /**< left hand side which led to old locks */
   SCIP_Real const       rhs                 /**< right hand side which led to old locks */
   )
{
   SCIP_VAR** vars;
   int nvars;
   SCIP_VAR* res;
   SCIP_Bool haslhs;
   SCIP_Bool hasrhs;
   int v;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(consanddata != NULL);
   assert(!SCIPisInfinity(scip, coef) && !SCIPisInfinity(scip, -coef));
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   /* choose correct variable array to add locks for, we only add locks for now valid variables */
   if( consanddata->nnewvars > 0 )
   {
      vars = consanddata->newvars;
      nvars = consanddata->nnewvars;
   }
   else
   {
      vars = consanddata->vars;
      nvars = consanddata->nvars;
   }

#ifndef NDEBUG
   if( SCIPisAndConsSorted(scip, consanddata->cons) )
   {
      SCIP_VAR** andvars;
         
      assert(consanddata->cons != NULL);

      assert(nvars == SCIPgetNVarsAnd(scip, consanddata->cons));
      andvars = SCIPgetVarsAnd(scip, consanddata->cons);
         
      /* check that consanddata object is correct*/
      for( v = nvars - 1; v > 0; --v )
         assert(vars[v] == andvars[v]);
   }
#endif

   res = SCIPgetResultantAnd(scip, consanddata->cons);
   assert(nvars == 0 || (vars != NULL && res != NULL));

   /* check which sites are infinity */
   haslhs = !SCIPisInfinity(scip, -lhs);
   hasrhs = !SCIPisInfinity(scip, rhs);

   if( SCIPconsIsLocked(cons) )
   {
      /* locking variables */
      if( SCIPisPositive(scip, coef) )
      {
         for( v = nvars - 1; v >= 0; --v )
         {
            SCIP_CALL( SCIPlockVarCons(scip, vars[v], cons, haslhs, hasrhs) );
         }
      }
      else
      {
         for( v = nvars - 1; v >= 0; --v )
         {
            SCIP_CALL( SCIPlockVarCons(scip, vars[v], cons, hasrhs, haslhs) );
         }
      }
      SCIP_CALL( SCIPlockVarCons(scip, res, cons, TRUE, TRUE) );
   }

   return SCIP_OKAY;
}

/** removes rounding locks for the given and-constraint associated with given coefficient */
static
SCIP_RETCODE unlockRoundingAndCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   CONSANDDATA*const     consanddata,        /**< CONSANDDATA object for which we want to delete the locks and the
                                              *   capture of the corresponding and-constraint */
   SCIP_Real const       coef,               /**< coefficient which led to old locks */
   SCIP_Real const       lhs,                /**< left hand side which led to old locks */
   SCIP_Real const       rhs                 /**< right hand side which led to old locks */
   )
{
   SCIP_VAR** vars;
   int nvars;
   SCIP_VAR* res;
   SCIP_Bool haslhs;
   SCIP_Bool hasrhs;
   int v;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(consanddata != NULL);
   assert(!SCIPisInfinity(scip, coef) && !SCIPisInfinity(scip, -coef));
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   vars = consanddata->vars;
   nvars = consanddata->nvars;

#ifndef NDEBUG
   if( consanddata->nnewvars == 0 && consanddata->cons != NULL && 
      SCIPconsIsActive(consanddata->cons) && SCIPisAndConsSorted(scip, consanddata->cons) )
   {
      SCIP_VAR** andvars;
      
      assert(nvars == SCIPgetNVarsAnd(scip, consanddata->cons));
      andvars = SCIPgetVarsAnd(scip, consanddata->cons);

      /* check that consanddata object is correct*/
      for( v = nvars - 1; v > 0; --v )
         assert(vars[v] == andvars[v]);
   }
#endif

   if( consanddata->cons != NULL )
      res = SCIPgetResultantAnd(scip, consanddata->cons);
   else
      res = NULL;
   assert(nvars == 0 || vars != NULL);

   /* check which sites are infinity */
   haslhs = !SCIPisInfinity(scip, -lhs);
   hasrhs = !SCIPisInfinity(scip, rhs);

   if( SCIPconsIsLocked(cons) )
   {
      /* unlock variables */
      if( SCIPisPositive(scip, coef) )
      {
         for( v = nvars - 1; v >= 0; --v )
         {
            //printf("unlocking var <%s>\n", SCIPvarGetName(vars[v]));
            SCIP_CALL( SCIPunlockVarCons(scip, vars[v], cons, haslhs, hasrhs) );
         }
      }
      else
      {
         for( v = nvars - 1; v >= 0; --v )
         {
            //printf("unlocking var <%s>\n", SCIPvarGetName(vars[v]));
            SCIP_CALL( SCIPunlockVarCons(scip, vars[v], cons, hasrhs, haslhs) );
         }
      }
      if( res != NULL )
      {
         //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(res), 1, 1);
         SCIP_CALL( SCIPunlockVarCons(scip, res, cons, TRUE, TRUE) );
      }
   }

   return SCIP_OKAY;
}

/** gets variables and coefficient of linear constraint */
static
SCIP_RETCODE getLinearConsVarsData(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< linear constraint */
   SCIP_LINEARCONSTYPE const constype,       /**< linear constraint type */
   SCIP_VAR**const       vars,               /**< array to store sorted (after indices) variables of linear constraint */
   SCIP_Real*const       coefs,              /**< array to store coefficient of linear constraint */
   int*const             nvars               /**< pointer to store number variables of linear constraint */
   )
{
   SCIP_VAR** linvars;
   int v;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(vars != NULL);
   assert(coefs != NULL);
   assert(nvars != NULL);

   /* determine for each special linear constrait all variables and coefficients */
   switch( constype )
   {
   case SCIP_LINEAR:
   {
      SCIP_Real* lincoefs;

      *nvars = SCIPgetNVarsLinear(scip, cons);
      linvars = SCIPgetVarsLinear(scip, cons);
      lincoefs = SCIPgetValsLinear(scip, cons);

      for( v = 0; v < *nvars; ++v )
      {
         vars[v] = linvars[v];
         coefs[v] = lincoefs[v];
      }
      break;
   }
   case SCIP_LOGICOR:
      *nvars = SCIPgetNVarsLogicor(scip, cons);
      linvars = SCIPgetVarsLogicor(scip, cons);

      for( v = 0; v < *nvars; ++v )
      {
         vars[v] = linvars[v];
         coefs[v] = 1.0;
      }
      break;
   case SCIP_KNAPSACK:
   {
      SCIP_Longint* weights;

      *nvars = SCIPgetNVarsKnapsack(scip, cons);
      linvars = SCIPgetVarsKnapsack(scip, cons);
      weights = SCIPgetWeightsKnapsack(scip, cons);

      for( v = 0; v < *nvars; ++v )
      {
         vars[v] = linvars[v];
         coefs[v] = (SCIP_Real) weights[v];
      }
      break;
   }
   case SCIP_SETPPC:
      *nvars = SCIPgetNVarsSetppc(scip, cons);
      linvars = SCIPgetVarsSetppc(scip, cons);

      for( v = 0; v < *nvars; ++v )
      {
         vars[v] = linvars[v];
         coefs[v] = 1.0;
      }
      break;
#if 0
   case SCIP_EQKNAPSACK:
   {
      SCIP_Longint* weights;

      *nvars = SCIPgetNVarsEQKnapsack(scip, cons);
      linvars = SCIPgetVarsEQKnapsack(scip, cons);
      weights = SCIPgetWeightsEQKnapsack(scip, cons);

      for( v = 0; v < *nvars; ++v )
      {
         vars[v] = linvars[v];
         coefs[v] = (SCIP_Real) weights[v];
      }
      break;
   }
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   /* sort variables after indices */
   SCIPsortPtr((void**)vars, SCIPvarComp, *nvars);

   return SCIP_OKAY;
}

/* calculate all not artificial linear variables and all artificial and-resultants which will be ordered like the
 * 'consanddatas' such that the and-resultant of the and-constraint is the and-resultant in the 'andress' array
 * afterwards 
 */
static
SCIP_RETCODE getLinVarsAndAndRess(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_VAR**const       vars,               /**< all variables of linear constraint */
   SCIP_Real*const       coefs,              /**< all coefficients of linear constraint */
   int const             nvars,              /**< number of all variables of linear constraint */
   SCIP_VAR**const       linvars,            /**< array to store not and-resultant variables of linear constraint, or NULL */
   SCIP_Real*const       lincoefs,           /**< array to store coefficients of not and-resultant variables of linear
                                              *   constraint, or NULL */
   int*const             nlinvars,           /**< pointer to store number of not and-resultant variables, or NULL */
   SCIP_VAR**const       andress,            /**< array to store and-resultant variables of linear constraint, or NULL */
   SCIP_Real*const       andcoefs,           /**< array to store coefficients of and-resultant variables of linear
                                              *   constraint, or NULL */
   int*const             nandress            /**< pointer to store number of and-resultant variables, or NULL */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   int v;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(vars != NULL);
   assert(coefs != NULL);
   assert((linvars != NULL) == ((lincoefs != NULL) && (nlinvars != NULL)));
   assert((andress != NULL) == ((andcoefs != NULL) && (nandress != NULL)));
   assert(linvars != NULL || andress != NULL);
   
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   if( nlinvars != NULL ) 
      *nlinvars = 0;
   if( nandress != NULL ) 
      *nandress = 0;

   conshdlr = SCIPconsGetHdlr(cons);
   assert(conshdlr != NULL);
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->hashmap != NULL);

   /* @Note: it is necessary that the linear constraint is merged (not needed for negated variables) and sorted after
    *        indices
    */

#ifndef NDEBUG
      /* check that old variables are sorted */
      for( v = nvars - 1; v > 0; --v )
         assert(SCIPvarGetIndex(vars[v]) > SCIPvarGetIndex(vars[v - 1]));
#endif
  
   /* split variables into original and artificial variables */
   for( v = 0; v < nvars; ++v )
   { 
      SCIP_Bool hashmapentryexists;

      assert(vars[v] != NULL);

      hashmapentryexists = SCIPhashmapExists(conshdlrdata->hashmap, (void*)(vars[v]));

      if( !hashmapentryexists && linvars != NULL ) // ????????? strstr(SCIPvarGetName(vars[v]), ARTIFICIALVARNAMEPREFIX) == NULL )
      {
         linvars[*nlinvars] = vars[v];
         lincoefs[*nlinvars] = coefs[v];
         ++(*nlinvars);
      }
      else if( hashmapentryexists && andress != NULL )
      {
         andress[*nandress] = vars[v];
         andcoefs[*nandress] = coefs[v];
         ++(*nandress);
      }
   }

   return SCIP_OKAY;
}

/** prints pseudoboolean constraint in CIP format to file stream */
static
SCIP_RETCODE consdataPrint(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   FILE*const            file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;

   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;
   SCIP_Real lhs;
   SCIP_Real rhs;

   SCIP_VAR** linvars;
   SCIP_Real* lincoefs;
   int nlinvars;
   int v;

   SCIP_VAR** andress;
   SCIP_Real* andcoefs;
   int nandress;
   
   SCIP_Bool printed;
   
   assert(scip != NULL);
   assert(cons != NULL);

   if( SCIPconsIsDeleted(cons) )
      return SCIP_OKAY;
   
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->lincons != NULL);
   /* more than one and-constraint is needed, otherwise this pseudoboolean constraint should be upgraded to a linear constraint */
   assert(consdata->nconsanddatas >= 0);

   /* gets number of variables in linear constraint */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );

   /* allocate temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andress, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nvars) );

   /* get sides of linear constraint */
   SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &lhs, &rhs) );
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   /* get variables and coefficient of linear constraint */
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
   assert(nvars == 0 || (vars != NULL && coefs != NULL));

   /* number of variables should be consistent, number of 'real' linear variables plus number of and-constraints should
    * have to be equal to the number of variables in the linear constraint
    */
   assert(consdata->nlinvars + consdata->nconsanddatas == nvars);

   /* print left hand side for ranged rows */
   if( !SCIPisInfinity(scip, -lhs)
      && !SCIPisInfinity(scip, rhs)
      && !SCIPisEQ(scip, lhs, rhs) )
      SCIPinfoMessage(scip, file, "%.15g <= ", lhs);

   nlinvars = 0;

   conshdlr = SCIPconsGetHdlr(cons);
   assert(conshdlr != NULL);
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->hashmap != NULL);
   
   nandress = 0;

   /* split variables into original and artificial variables */
   for( v = 0; v < nvars; ++v )
   { 
      assert(vars[v] != NULL);

      if( !SCIPhashmapExists(conshdlrdata->hashmap, (void*)(vars[v])) ) // ????????? strstr(SCIPvarGetName(vars[v]), ARTIFICIALVARNAMEPREFIX) == NULL )
      {
         linvars[nlinvars] = vars[v];
         lincoefs[nlinvars] = coefs[v];
         ++nlinvars;
      }
      else
      {
         andress[nandress] = vars[v];
         andcoefs[nandress] = coefs[v];
         ++nandress;
      }
   }
   assert(nandress == consdata->nconsanddatas);

   printed= FALSE;

   /* print coefficients and variables */
   if( nlinvars > 0)
   {
      printed= TRUE;

      /* print linear part of constraint */
      SCIP_CALL( SCIPwriteVarsLinearsum(scip, file, linvars, lincoefs, nlinvars, TRUE) );
   }

   

   for( v = nandress - 1; v >= 0; --v )
   {
      CONSANDDATA* consanddata;
      SCIP_CONS* andcons;
      SCIP_VAR** andvars;
      int nandvars;

      /* if the and resultant was fixed we print a constant */
      if( SCIPvarGetLbLocal(andress[v]) > 0.5 || SCIPvarGetUbLocal(andress[v]) < 0.5 )
      {
         if( SCIPvarGetLbLocal(andress[v]) > 0.5 )
         {
            printed = TRUE;
            SCIPinfoMessage(scip, file, " %+.15g ", andcoefs[v] * SCIPvarGetLbLocal(andress[v]));
         }
         continue;
      }
      else if( SCIPvarGetStatus(andress[v]) == SCIP_VARSTATUS_AGGREGATED )
      {
         SCIP_VAR* aggrvar;
         SCIP_Bool negated;

         SCIP_CALL( SCIPgetBinvarRepresentative(scip, andress[v], &aggrvar, &negated) );
         assert(aggrvar != NULL);
         assert(SCIPvarGetType(aggrvar) == SCIP_VARTYPE_BINARY);

         printed = TRUE;
         SCIPinfoMessage(scip, file, " %+.15g <%s>[B]", andcoefs[v], SCIPvarGetName(aggrvar));

         continue;
      }

      consanddata = (CONSANDDATA*) SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)andress[v]);
      assert(consanddata != NULL);

      if( SCIPconsIsOriginal(cons) )
      {
         andcons = consanddata->origcons;
         /* if problem was not yet transformed, origcons is not yet initialized */
         if( andcons == NULL )
            andcons = consanddata->cons;
      }
      else
         andcons = consanddata->cons;
      assert(andcons != NULL);

      andvars = SCIPgetVarsAnd(scip, andcons);
      nandvars = SCIPgetNVarsAnd(scip, andcons);
      assert(nandvars == 0 || andvars != NULL);

      if( nandvars > 0 )
      {
         printed = TRUE;
         SCIPinfoMessage(scip, file, " %+.15g ", andcoefs[v]);

         /* @todo: better write new method SCIPwriteProduct */
         /* print variable list */
         SCIP_CALL( SCIPwriteVarsList(scip, file, andvars, nandvars, TRUE) );
      }
   }
   
   if( !printed )
   {
      SCIPinfoMessage(scip, file, " 0 ");
   }

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &andcoefs);
   SCIPfreeBufferArray(scip, &andress);
   SCIPfreeBufferArray(scip, &lincoefs);
   SCIPfreeBufferArray(scip, &linvars);
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   /* print right hand side */
   if( SCIPisEQ(scip, lhs, rhs) )
      SCIPinfoMessage(scip, file, "== %.15g", rhs);
   else if( !SCIPisInfinity(scip, rhs) )
      SCIPinfoMessage(scip, file, "<= %.15g", rhs);
   else if( !SCIPisInfinity(scip, -lhs) )
      SCIPinfoMessage(scip, file, ">= %.15g", lhs);
   else
      SCIPinfoMessage(scip, file, " [free]");

   return SCIP_OKAY;
}

/* creates and/or adds the resultant for a given term */
static
SCIP_RETCODE createAndAddAndCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*const   conshdlr,           /**< pseudoboolean constraint handler */
   SCIP_VAR**const       vars,               /**< array of variables to get and-constraints for */
   int const             nvars,              /**< number of variables to get and-constraints for */
   SCIP_Bool const       initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                              *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool const       enforce,            /**< should the constraint be enforced during node processing?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool const       check,              /**< should the constraint be checked for feasibility?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool const       local,              /**< is constraint only valid locally?
                                              *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool const       modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool const       dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which 
                                              *   are seperated as constraints. */
   SCIP_Bool const       stickingatnode,     /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   SCIP_CONS**const      andcons             /**< pointer to store and-constraint */
   )
{
   CONSANDDATA* newdata;
   CONSANDDATA* tmpdata;
   SCIP_CONSHDLRDATA* conshdlrdata;
   char name[SCIP_MAXSTRLEN];
   SCIP_Bool separate;
   SCIP_Bool propagate;
   SCIP_Bool removable;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(vars != NULL);
   assert(nvars > 0);
   assert(andcons != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->hashtable != NULL);

   /* allocate memory for a possible new consanddata object */
   SCIP_CALL( SCIPallocBlockMemory(scip, &newdata) );
   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(newdata->vars), vars, nvars) );
   newdata->nvars = nvars;
   newdata->svars = nvars;
   newdata->newvars = NULL;
   newdata->nnewvars = 0;
   newdata->snewvars = 0;
   newdata->nuses = 0;
   newdata->deleted = FALSE;
   newdata->cons = NULL;
   newdata->origcons = NULL;

   /* sort variables */
   SCIPsortPtr((void**)(newdata->vars), SCIPvarComp, nvars);

   /* get constraint from current hash table with same variables as cons0 */
   tmpdata = (CONSANDDATA*)(SCIPhashtableRetrieve(conshdlrdata->hashtable, (void*)newdata));

   /* if there is already the same and constraint created use this resultant */
   if( tmpdata != NULL )
   {
#ifndef NDEBUG
      SCIP_VAR* res;

      assert(tmpdata->cons != NULL);
      res = SCIPgetResultantAnd(scip, tmpdata->cons);
      assert(res != NULL);

      /* check that we already have added this resultant to and-constraint entry */
      assert(SCIPhashmapExists(conshdlrdata->hashmap, (void*)res));
#endif
      *andcons = tmpdata->cons;
      assert(*andcons != NULL);

      /* increase usage of data object */
      ++(tmpdata->nuses);
   }
   else
   { 
      /* create new and-constraint */
      SCIP_CONS* newcons;
      SCIP_VAR* resultant;

      /* create auxiliary variable */
      (void)SCIPsnprintf(name, SCIP_MAXSTRLEN, ARTIFICIALVARNAMEPREFIX"%d", conshdlrdata->nallconsanddatas);
      SCIP_CALL( SCIPcreateVar(scip, &resultant, name, 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY, 
            TRUE, TRUE, NULL, NULL, NULL, NULL, NULL) );

      /* @todo: check whether we want to branch on artificial variables */
#if 1
      /* change branching priority of artificial variable to -1 */
      SCIP_CALL( SCIPchgVarBranchPriority(scip, resultant, -1) );
#endif
 
      /* add auxiliary variable to the problem */
      SCIP_CALL( SCIPaddVar(scip, resultant) );

      SCIP_CALL( SCIPgetBoolParam(scip, "constraints/"CONSHDLR_NAME"/nlcseparate", &separate) );
      SCIP_CALL( SCIPgetBoolParam(scip, "constraints/"CONSHDLR_NAME"/nlcpropagate", &propagate) );
      SCIP_CALL( SCIPgetBoolParam(scip, "constraints/"CONSHDLR_NAME"/nlcremovable", &removable) );

      /* we do not want to check the and constraints, so the check flag will be FALSE */
            
      /* create and add "and" constraint for the multiplication of the binary variables */ 
      (void)SCIPsnprintf(name, SCIP_MAXSTRLEN, "andcons_%d", conshdlrdata->nallconsanddatas);
      SCIP_CALL( SCIPcreateConsAnd(scip, &newcons, name, resultant, newdata->nvars, newdata->vars,
            initial, separate, enforce, FALSE, propagate, 
            local, modifiable, dynamic, removable, stickingatnode) );
      SCIP_CALL( SCIPaddCons(scip, newcons) );
      SCIPdebug( SCIP_CALL( SCIPprintCons(scip, newcons, NULL) ) );

      *andcons = newcons;
      assert(*andcons != NULL);

      /* resize data for all and-constraints if necessary */
      if( conshdlrdata->nallconsanddatas == conshdlrdata->sallconsanddatas )
      {
         SCIP_CALL( SCIPensureBlockMemoryArray(scip, &(conshdlrdata->allconsanddatas), &(conshdlrdata->sallconsanddatas), SCIPcalcMemGrowSize(scip, conshdlrdata->sallconsanddatas + 1)) );
      }

      /* add new data object to global hash table */
      conshdlrdata->allconsanddatas[conshdlrdata->nallconsanddatas] = newdata;
      ++(conshdlrdata->nallconsanddatas);

      /* increase usage of data object */
      ++(newdata->nuses);
      
      newdata->cons = newcons;
      SCIP_CALL( SCIPcaptureCons(scip, newdata->cons) );

      /* no such and-constraint in current hash table: insert the new object into hash table */  
      SCIP_CALL( SCIPhashtableInsert(conshdlrdata->hashtable, (void*)newdata) );

      /* insert new mapping */
      assert(!SCIPhashmapExists(conshdlrdata->hashmap, (void*)resultant));
      SCIP_CALL( SCIPhashmapInsert(conshdlrdata->hashmap, (void*)resultant, (void*)newdata) );

      /* release and-resultant and -constraint */
      SCIP_CALL( SCIPreleaseVar(scip, &resultant) );
      SCIP_CALL( SCIPreleaseCons(scip, &newcons) );

      return SCIP_OKAY;
   }

   /* free memory */
   SCIPfreeBlockMemoryArray(scip, &(newdata->vars), newdata->svars);
   SCIPfreeBlockMemory(scip, &newdata);

   return SCIP_OKAY;
}

/** create an and constraint and adds it to the problem and to the linear constraint */
static
SCIP_RETCODE addCoefTerm(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_VAR**const       vars,               /**< variables of the nonlinear term */
   int const             nvars,              /**< number of variables of the nonlinear term */
   SCIP_Real const       val                 /**< coefficient of constraint entry */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONS* andcons;
   SCIP_CONSDATA* consdata;
   SCIP_VAR* res;
   
   assert(scip != NULL);
   assert(cons != NULL);
   assert(nvars == 0 || vars != NULL);

   if( nvars == 0 || SCIPisZero(scip, val) )
      return SCIP_OKAY;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   conshdlr = SCIPconsGetHdlr(cons);
   assert(conshdlr != NULL);

   conshdlrdata =  SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* create (and add) and-constraint */
   SCIP_CALL( createAndAddAndCons(scip, conshdlr, vars, nvars,
         SCIPconsIsInitial(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons), SCIPconsIsLocal(cons),
         SCIPconsIsModifiable(cons), SCIPconsIsDynamic(cons), SCIPconsIsStickingAtNode(cons), 
         &andcons) );
   assert(andcons != NULL);

   /* ensure memory size */
   if( consdata->nconsanddatas == consdata->sconsanddatas )
   {
      SCIP_CALL( SCIPensureBlockMemoryArray(scip, &(consdata->consanddatas), &(consdata->sconsanddatas), consdata->sconsanddatas + 1) );
   }

   res = SCIPgetResultantAnd(scip, andcons);
   assert(res != NULL);
   assert(SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res) != NULL);

   consdata->consanddatas[consdata->nconsanddatas] = (CONSANDDATA*) SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res);
   ++(consdata->nconsanddatas);

   /* add auxiliary variables to linear constraint */
   switch( consdata->linconstype )
   {
   case SCIP_LINEAR:
      SCIP_CALL( SCIPaddCoefLinear(scip, consdata->lincons, res, val) );
      break;
   case SCIP_LOGICOR:
      if( !SCIPisEQ(scip, val, 1.0) )
         return SCIP_INVALIDDATA;

      SCIP_CALL( SCIPaddCoefLogicor(scip, consdata->lincons, res) );
      break;
   case SCIP_KNAPSACK:
      if( !SCIPisIntegral(scip, val) || !SCIPisPositive(scip, val) )
         return SCIP_INVALIDDATA;

      SCIP_CALL( SCIPaddCoefKnapsack(scip, consdata->lincons, res, (SCIP_Longint) val) );
      break;
   case SCIP_SETPPC:
      if( !SCIPisEQ(scip, val, 1.0) )
         return SCIP_INVALIDDATA;
      
      SCIP_CALL( SCIPaddCoefSetppc(scip, consdata->lincons, res) );
      break;
#if 0
   case SCIP_EQKNAPSACK:
      if( !SCIPisIntegral(scip, val) || !SCIPisPositive(scip, val) )
         return SCIP_INVALIDDATA;

      SCIP_CALL( SCIPaddCoefEQKnapsack(scip, consdata->lincons, res, (SCIP_Longint) val) );
      break;
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   /* install rounding locks for all new variable */
   SCIP_CALL( lockRoundingAndCons(scip, cons, consdata->consanddatas[consdata->nconsanddatas - 1], val, consdata->lhs, consdata->rhs) );

   /* change flags */
   consdata->changed = TRUE;
   consdata->propagated = FALSE;
   consdata->presolved = FALSE;
   consdata->cliquesadded = FALSE;
   consdata->upgradetried = FALSE;

   return SCIP_OKAY;
}

/** changes left hand side of linear constraint */
static
SCIP_RETCODE chgLhsLinearCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< linear constraint */
   SCIP_LINEARCONSTYPE const constype,       /**< linear constraint type */
   SCIP_Real const       lhs                 /**< new left hand side of linear constraint */
   )
{
   switch( constype )
   {
   case SCIP_LINEAR:
      SCIP_CALL( SCIPchgLhsLinear(scip, cons, lhs) );
   case SCIP_LOGICOR:
   case SCIP_KNAPSACK:
   case SCIP_SETPPC:
      SCIPerrorMessage("changing left hand side only allowed on standard lienar constraint \n");
      return SCIP_INVALIDDATA;
#if 0
   case SCIP_EQKNAPSACK:
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** changes right hand side of linear constraint */
static
SCIP_RETCODE chgRhsLinearCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< linear constraint */
   SCIP_LINEARCONSTYPE const constype,       /**< linear constraint type */
   SCIP_Real const       rhs                 /**< new right hand side of linear constraint */
   )
{
   switch( constype )
   {
   case SCIP_LINEAR:
      SCIP_CALL( SCIPchgRhsLinear(scip, cons, rhs) );
   case SCIP_LOGICOR:
   case SCIP_KNAPSACK:
   case SCIP_SETPPC:
      SCIPerrorMessage("changing left hand side only allowed on standard lienar constraint \n");
      return SCIP_INVALIDDATA;
#if 0
   case SCIP_EQKNAPSACK:
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** sets left hand side of linear constraint */
static
SCIP_RETCODE chgLhs(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< linear constraint */
   SCIP_Real             lhs                 /**< new left hand side */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;
   SCIP_VAR** linvars;
   SCIP_Real* lincoefs;
   int nlinvars;
   SCIP_VAR** andress;
   SCIP_Real* andcoefs;
   int nandress;
   SCIP_Real oldlhs;
   SCIP_Real oldrhs;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(!SCIPisInfinity(scip, lhs));

   /* adjust value to not be smaller than -inf */
   if ( SCIPisInfinity(scip, -lhs) )
      lhs = -SCIPinfinity(scip);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* get sides of linear constraint */
   SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &oldlhs, &oldrhs) );
   assert(!SCIPisInfinity(scip, oldlhs));
   assert(!SCIPisInfinity(scip, -oldrhs));
   assert(SCIPisLE(scip, oldlhs, oldrhs));

   /* check whether the side is not changed */
   if( SCIPisEQ(scip, oldlhs, lhs) )
      return SCIP_OKAY;

   /* gets number of variables in linear constraint */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );

   /* allocate temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andress, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nvars) );

   /* get variables and coefficient of linear constraint */
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
   assert(nvars == 0 || (vars != NULL && coefs != NULL));

   /* calculate all not artificial linear variables and all artificial and-resultants which will be ordered like the
    * 'consanddatas' such that the and-resultant of the and-constraint is the and-resultant in the 'andress' array
    * afterwards 
    */
   SCIP_CALL( getLinVarsAndAndRess(scip, cons, vars, coefs, nvars, linvars, lincoefs, &nlinvars, andress, andcoefs, &nandress) );
   
   /* if necessary, update the rounding locks of variables */
   if( SCIPconsIsLocked(cons) )
   {
      SCIP_VAR** andvars;
      int nandvars;
      SCIP_Real val;
      int v;
      int c;

      assert(SCIPconsIsTransformed(cons));

      if( SCIPisInfinity(scip, -oldlhs) && !SCIPisInfinity(scip, -lhs) )
      {
#if 0
         SCIP_VAR* var;

         /* linear part */
         for( v = nlinvars - 1; v >= 0; --v )
         {
            var = linvars[v];
            val = lincoefs[v];

            /* lock variable */
            if( SCIPisPositive(scip, val) )
            {
               SCIP_CALL( SCIPlockVarCons(scip, var, cons, TRUE, FALSE) );
            }
            else
            {
               SCIP_CALL( SCIPlockVarCons(scip, var, cons, FALSE, TRUE) );
            }
         }
#endif
         /* non-linear part */
         for( c = consdata->nconsanddatas - 1; c >= 0; --c )
         {
            CONSANDDATA* consanddata;
            SCIP_CONS* andcons;
            
            consanddata = consdata->consanddatas[c];
            assert(consanddata != NULL);

            andcons = consanddata->cons;
            assert(andcons != NULL);

            andvars = SCIPgetVarsAnd(scip, andcons);
            nandvars = SCIPgetNVarsAnd(scip, andcons);
            val = andcoefs[v];

            /* lock variables */
            if( SCIPisPositive(scip, val) )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPlockVarCons(scip, andvars[v], cons, TRUE, FALSE) );
               }
            }
            else
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPlockVarCons(scip, andvars[v], cons, FALSE, TRUE) );
               }
            }
         }
      }
      else if( !SCIPisInfinity(scip, -oldlhs) && SCIPisInfinity(scip, -lhs) )
      {
#if 0
         SCIP_VAR* var;

         /* linear part */
         for( v = nlinvars - 1; v >= 0; --v )
         {
            var = linvars[v];
            val = lincoefs[v];

            /* unlock variable */
            if( SCIPisPositive(scip, val) )
            {
               SCIP_CALL( SCIPunlockVarCons(scip, var, cons, TRUE, FALSE) );
            }
            else
            {
               SCIP_CALL( SCIPunlockVarCons(scip, var, cons, FALSE, TRUE) );
            }
         }
#endif

         /* non-linear part */
         for( c = consdata->nconsanddatas - 1; c >= 0; --c )
         {
            CONSANDDATA* consanddata;
            SCIP_CONS* andcons;
            
            consanddata = consdata->consanddatas[c];
            assert(consanddata != NULL);

            andcons = consanddata->cons;
            assert(andcons != NULL);

            andvars = SCIPgetVarsAnd(scip, andcons);
            nandvars = SCIPgetNVarsAnd(scip, andcons);
            val = andcoefs[v];

            /* lock variables */
            if( SCIPisPositive(scip, val) )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPunlockVarCons(scip, andvars[v], cons, TRUE, FALSE) );
               }
            }
            else
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPunlockVarCons(scip, andvars[v], cons, FALSE, TRUE) );
               }
            }
         }
      }
   }

   /* check whether the left hand side is increased, if and only if that's the case we maybe can propagate, tighten and add more cliques */
   if( SCIPisLT(scip, oldlhs, lhs) )
   {
      consdata->propagated = FALSE;
   }

   /* set new left hand side and update constraint data */
   SCIP_CALL( chgLhsLinearCons(scip, consdata->lincons, consdata->linconstype, lhs) );
   consdata->lhs = lhs;
   consdata->presolved = FALSE;
   consdata->changed = TRUE;

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &andcoefs);
   SCIPfreeBufferArray(scip, &andress);
   SCIPfreeBufferArray(scip, &lincoefs);
   SCIPfreeBufferArray(scip, &linvars);
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/** sets right hand side of pseudoboolean constraint */
static
SCIP_RETCODE chgRhs(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< linear constraint */
   SCIP_Real             rhs                 /**< new right hand side */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;
   SCIP_VAR** linvars;
   SCIP_Real* lincoefs;
   int nlinvars;
   SCIP_VAR** andress;
   SCIP_Real* andcoefs;
   int nandress;
   SCIP_Real oldlhs;
   SCIP_Real oldrhs;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(!SCIPisInfinity(scip, -rhs));

   /* adjust value to not be larger than inf */
   if ( SCIPisInfinity(scip, rhs) )
      rhs = SCIPinfinity(scip);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* get sides of linear constraint */
   SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &oldlhs, &oldrhs) );
   assert(!SCIPisInfinity(scip, oldlhs));
   assert(!SCIPisInfinity(scip, -oldrhs));
   assert(SCIPisLE(scip, oldlhs, oldrhs));

   /* check whether the side is not changed */
   if( SCIPisEQ(scip, oldrhs, rhs) )
      return SCIP_OKAY;

   /* gets number of variables in linear constraint */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );

   /* allocate temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andress, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nvars) );

   /* get variables and coefficient of linear constraint */
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
   assert(nvars == 0 || (vars != NULL && coefs != NULL));

   /* calculate all not artificial linear variables and all artificial and-resultants which will be ordered like the
    * 'consanddatas' such that the and-resultant of the and-constraint is the and-resultant in the 'andress' array
    * afterwards 
    */
   SCIP_CALL( getLinVarsAndAndRess(scip, cons, vars, coefs, nvars, linvars, lincoefs, &nlinvars, andress, andcoefs, &nandress) );

   /* if necessary, update the rounding locks of variables */
   if( SCIPconsIsLocked(cons) )
   {
      SCIP_VAR** andvars;
      int nandvars;
      SCIP_Real val;
      int v;
      int c;

      assert(SCIPconsIsTransformed(cons));

      if( SCIPisInfinity(scip, oldrhs) && !SCIPisInfinity(scip, rhs) )
      {
#if 0
         SCIP_VAR* var;

         /* linear part */
         for( v = nlinvars - 1; v >= 0; --v )
         {
            var = linvars[v];
            val = lincoefs[v];

            /* lock variable */
            if( SCIPisPositive(scip, val) )
            {
               SCIP_CALL( SCIPlockVarCons(scip, var, cons, FALSE, TRUE) );
            }
            else
            {
               SCIP_CALL( SCIPlockVarCons(scip, var, cons, TRUE, FALSE) );
            }
         }
#endif

         /* non-linear part */
         for( c = consdata->nconsanddatas - 1; c >= 0; --c )
         {
            CONSANDDATA* consanddata;
            SCIP_CONS* andcons;
            
            consanddata = consdata->consanddatas[c];
            assert(consanddata != NULL);

            andcons = consanddata->cons;
            assert(andcons != NULL);

            andvars = SCIPgetVarsAnd(scip, andcons);
            nandvars = SCIPgetNVarsAnd(scip, andcons);
            val = andcoefs[v];

            /* lock variables */
            if( SCIPisPositive(scip, val) )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPlockVarCons(scip, andvars[v], cons, FALSE, TRUE) );
               }
            }
            else
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPlockVarCons(scip, andvars[v], cons, TRUE, FALSE) );
               }
            }
         }
      }
      else if( !SCIPisInfinity(scip, oldrhs) && SCIPisInfinity(scip, rhs) )
      {
#if 0
         SCIP_VAR* var;

         /* linear part */
         for( v = nlinvars - 1; v >= 0; --v )
         {
            var = linvars[v];
            val = lincoefs[v];

            /* unlock variable */
            if( SCIPisPositive(scip, val) )
            {
               SCIP_CALL( SCIPunlockVarCons(scip, var, cons, FALSE, TRUE) );
            }
            else
            {
               SCIP_CALL( SCIPunlockVarCons(scip, var, cons, TRUE, FALSE) );
            }
         }
#endif

         /* non-linear part */
         for( c = consdata->nconsanddatas - 1; c >= 0; --c )
         {
            CONSANDDATA* consanddata;
            SCIP_CONS* andcons;
            
            consanddata = consdata->consanddatas[c];
            assert(consanddata != NULL);

            andcons = consanddata->cons;
            assert(andcons != NULL);

            andvars = SCIPgetVarsAnd(scip, andcons);
            nandvars = SCIPgetNVarsAnd(scip, andcons);
            val = andcoefs[v];

            /* lock variables */
            if( SCIPisPositive(scip, val) )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPunlockVarCons(scip, andvars[v], cons, FALSE, TRUE) );
               }
            }
            else
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  SCIP_CALL( SCIPunlockVarCons(scip, andvars[v], cons, TRUE, FALSE) );
               }
            }
         }
      }
   }

   /* check whether the right hand side is decreased, if and only if that's the case we maybe can propagate, tighten and add more cliques */
   if( SCIPisGT(scip, oldrhs, rhs) )
   {
      consdata->propagated = FALSE;
   }

   /* set new right hand side and update constraint data */
   SCIP_CALL( chgRhsLinearCons(scip, consdata->lincons, consdata->linconstype, rhs) );
   consdata->rhs = rhs;
   consdata->presolved = FALSE;
   consdata->changed = TRUE;

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &andcoefs);
   SCIPfreeBufferArray(scip, &andress);
   SCIPfreeBufferArray(scip, &lincoefs);
   SCIPfreeBufferArray(scip, &linvars);
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/* create and-constraints and get all and-resultants */
static
SCIP_RETCODE createAndAddAnds(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*const   conshdlr,           /**< pseudoboolean constraint handler */
   SCIP_VAR**const*const terms,              /**< array of term variables to get and-constraints for */
   SCIP_Real*const       termcoefs,          /**< array of coefficients for and-constraints */
   int const             nterms,             /**< number of terms to get and-constraints for */
   int const*const       ntermvars,          /**< array of number of variable in each term */
   SCIP_Bool const       initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                              *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool const       enforce,            /**< should the constraint be enforced during node processing?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool const       check,              /**< should the constraint be checked for feasibility?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool const       local,              /**< is constraint only valid locally?
                                              *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool const       modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool const       dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which 
                                              *   are seperated as constraints. */
   SCIP_Bool const       stickingatnode,     /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   SCIP_CONS**const      andconss,           /**< array to store all created and-constraints for given terms */
   SCIP_Real*const       andvals,            /**< array to store all coefficients of and-constraints */
   int*const             nandconss           /**< number of created and constraints */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   int t;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(nterms == 0 || (terms != NULL && ntermvars != NULL));
   assert(andconss != NULL);
   assert(andvals != NULL);
   assert(nandconss != NULL);

   (*nandconss) = 0;

   if( nterms == 0 )
      return SCIP_OKAY;

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* loop over all terms and created/get all and constraints */
   for( t = 0; t < nterms; ++t )
   {
      if( !SCIPisZero(scip, termcoefs[t]) && ntermvars[t] > 0 )
      {
         SCIP_CALL( createAndAddAndCons(scip, conshdlr, terms[t], ntermvars[t],
               initial, enforce, check, local, modifiable, dynamic, stickingatnode, 
               &(andconss[*nandconss])) );
         assert(andconss[*nandconss] != NULL);
         andvals[*nandconss] = termcoefs[t];
         ++(*nandconss);
      }
   }

   return SCIP_OKAY;
}

/** created linear constraint of pseudo boolean constraint */
static
SCIP_RETCODE createAndAddLinearCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*const   conshdlr,           /**< pseudoboolean constraint handler */
   SCIP_VAR**const       linvars,            /**< linear variables */
   int const             nlinvars,           /**< number of linear variables */
   SCIP_Real*const       linvals,            /**< linear coefficients */
   SCIP_VAR**const       andress,            /**< and-resultant variables */
   int const             nandress,           /**< number of and-resultant variables */
   SCIP_Real const*const andvals,            /**< and-resultant coefficients */
   SCIP_Real const       lhs,                /**< left hand side of linear constraint */
   SCIP_Real const       rhs,                /**< right hand side of linear constraint */
   SCIP_Bool const       initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                              *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool const       separate,           /**< should the constraint be separated during LP processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool const       enforce,            /**< should the constraint be enforced during node processing?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool const       check,              /**< should the constraint be checked for feasibility?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool const       propagate,          /**< should the constraint be propagated during node processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool const       local,              /**< is constraint only valid locally?
                                              *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool const       modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool const       dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which 
                                              *   are seperated as constraints. */
   SCIP_Bool const       removable,          /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   SCIP_Bool const       stickingatnode,     /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   SCIP_CONS**const      lincons,            /**< pointer to store created linear constraint */
   SCIP_LINEARCONSTYPE*const linconstype     /**< pointer to store the type of the linear constraint */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSHDLR* upgrconshdlr;
   SCIP_CONS* cons;
   char name[SCIP_MAXSTRLEN];
   int v;
   SCIP_Bool created;
   SCIP_Bool integral;
   int nzero;
   int ncoeffspone;
   int ncoeffsnone;
   int ncoeffspint;
   int ncoeffsnint;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(nlinvars == 0 || (linvars != NULL && linvals != NULL));
   assert(nandress == 0 || (andress != NULL && andvals != NULL));
   assert(lincons != NULL);
   assert(linconstype != NULL);
   assert(nlinvars > 0 || nandress > 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   (*linconstype) = -1;
   (*lincons) = NULL;

   (void)SCIPsnprintf(name, SCIP_MAXSTRLEN, "pseudoboolean_linear%d", conshdlrdata->nlinconss);
   ++(conshdlrdata->nlinconss);

   created = FALSE;

   if( !modifiable )
   {
      SCIP_Real val;
      int nvars;

      /* calculate some statistics for upgrading on linear constraint */
      nzero = 0;
      ncoeffspone = 0;
      ncoeffsnone = 0;
      ncoeffspint = 0;
      ncoeffsnint = 0;
      integral = TRUE;
      nvars = nlinvars + nandress;

      /* calculate information over linear part */
      for( v = nlinvars - 1; v >= 0; --v )
      {
         val = linvals[v];
      
         if( SCIPisZero(scip, val) )
         {
            ++nzero;
            continue;
         }
         if( SCIPisEQ(scip, val, 1.0) )
            ++ncoeffspone;
         else if( SCIPisEQ(scip, val, -1.0) )
            ++ncoeffsnone;
         else if( SCIPisIntegral(scip, val) )
         {
            if( SCIPisPositive(scip, val) )
               ++ncoeffspint;
            else
               ++ncoeffsnint;
         }
         else
            integral = FALSE;
      }

      /* calculate information over and-resultants */
      for( v = nandress - 1; v >= 0; --v )
      {
         val = andvals[v];
      
         if( SCIPisZero(scip, val) )
         {
            ++nzero;
            continue;
         }
         if( SCIPisEQ(scip, val, 1.0) )
            ++ncoeffspone;
         else if( SCIPisEQ(scip, val, -1.0) )
            ++ncoeffsnone;
         else if( SCIPisIntegral(scip, val) )
         {
            if( SCIPisPositive(scip, val) )
               ++ncoeffspint;
            else
               ++ncoeffsnint;
         }
         else
            integral = FALSE;
      }
   
      upgrconshdlr = SCIPfindConshdlr(scip, "logicor");

      /* check, if linear constraint can be upgraded to logic or constraint
       * - logic or constraints consist only of binary variables with a
       *   coefficient of +1.0 or -1.0 (variables with -1.0 coefficients can be negated):
       *        lhs     <= x1 + ... + xp - y1 - ... - yn <= rhs
       * - negating all variables y = (1-Y) with negative coefficients gives:
       *        lhs + n <= x1 + ... + xp + Y1 + ... + Yn <= rhs + n
       * - negating all variables x = (1-X) with positive coefficients and multiplying with -1 gives:
       *        p - rhs <= X1 + ... + Xp + y1 + ... + yn <= p - lhs
       * - logic or constraints have left hand side of +1.0, and right hand side of +infinity: x(S) >= 1.0
       *    -> without negations:  (lhs == 1 - n  and  rhs == +inf)  or  (lhs == -inf  and  rhs = p - 1)
       */
      if( upgrconshdlr != NULL && nvars > 2 && ncoeffspone + ncoeffsnone == nvars
         && ((SCIPisEQ(scip, lhs, 1.0 - ncoeffsnone) && SCIPisInfinity(scip, rhs))
            || (SCIPisInfinity(scip, -lhs) && SCIPisEQ(scip, rhs, ncoeffspone - 1.0))) )
      {
         SCIP_VAR** transvars;
         int mult;

         SCIPdebugMessage("linear constraint will be logic-or constraint\n");
      
         /* check, if we have to multiply with -1 (negate the positive vars) or with +1 (negate the negative vars) */
         mult = SCIPisInfinity(scip, rhs) ? +1 : -1;
      
         /* get temporary memory */
         SCIP_CALL( SCIPallocBufferArray(scip, &transvars, nvars) );
      
         /* negate positive or negative variables */
         for( v = 0; v < nlinvars; ++v )
         {
            if( mult * linvals[v] > 0.0 )
               transvars[v] = linvars[v];
            else
            {
               SCIP_CALL( SCIPgetNegatedVar(scip, linvars[v], &transvars[v]) );
            }
            assert(transvars[v] != NULL);
         }

         /* negate positive or negative variables */
         for( v = 0; v < nandress; ++v )
         {
            if( mult * andvals[v] > 0.0 )
               transvars[nlinvars + v] = andress[v];
            else
            {
               SCIP_CALL( SCIPgetNegatedVar(scip, andress[v], &transvars[nlinvars + v]) );
            }
            assert(transvars[nlinvars + v] != NULL);
         }

         assert(!modifiable);
         /* create the constraint */
         SCIP_CALL( SCIPcreateConsLogicor(scip, &cons, name, nvars, transvars,
               initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );

         created = TRUE;
         (*linconstype) = SCIP_LOGICOR;

         /* free temporary memory */
         SCIPfreeBufferArray(scip, &transvars);
      }
   
      upgrconshdlr = SCIPfindConshdlr(scip, "setppc");

      /* check, if linear constraint can be upgraded to set partitioning, packing, or covering constraint
       * - all set partitioning / packing / covering constraints consist only of binary variables with a
       *   coefficient of +1.0 or -1.0 (variables with -1.0 coefficients can be negated):
       *        lhs     <= x1 + ... + xp - y1 - ... - yn <= rhs
       * - negating all variables y = (1-Y) with negative coefficients gives:
       *        lhs + n <= x1 + ... + xp + Y1 + ... + Yn <= rhs + n
       * - negating all variables x = (1-X) with positive coefficients and multiplying with -1 gives:
       *        p - rhs <= X1 + ... + Xp + y1 + ... + yn <= p - lhs
       * - a set partitioning constraint has left hand side of +1.0, and right hand side of +1.0 : x(S) == 1.0
       *    -> without negations:  lhs == rhs == 1 - n  or  lhs == rhs == p - 1
       * - a set packing constraint has left hand side of -infinity, and right hand side of +1.0 : x(S) <= 1.0
       *    -> without negations:  (lhs == -inf  and  rhs == 1 - n)  or  (lhs == p - 1  and  rhs = +inf)
       * - a set covering constraint has left hand side of +1.0, and right hand side of +infinity: x(S) >= 1.0
       *    -> without negations:  (lhs == 1 - n  and  rhs == +inf)  or  (lhs == -inf  and  rhs = p - 1)
       */
      if( upgrconshdlr != NULL && !created && ncoeffspone + ncoeffsnone == nvars )
      {
         SCIP_VAR** transvars;
         int mult;

         if( SCIPisEQ(scip, lhs, rhs) && (SCIPisEQ(scip, lhs, 1.0 - ncoeffsnone) || SCIPisEQ(scip, lhs, ncoeffspone - 1.0)) )
         {
            SCIPdebugMessage("linear pseudoboolean constraint will be a set partitioning constraint\n");

            /* check, if we have to multiply with -1 (negate the positive vars) or with +1 (negate the negative vars) */
            mult = SCIPisEQ(scip, lhs, 1.0 - ncoeffsnone) ? +1 : -1;

            /* get temporary memory */
            SCIP_CALL( SCIPallocBufferArray(scip, &transvars, nvars) );

            /* negate positive or negative variables for linear variables */
            for( v = 0; v < nlinvars; ++v )
            {
               if( mult * linvals[v] > 0.0 )
                  transvars[v] = linvars[v];
               else
               {
                  SCIP_CALL( SCIPgetNegatedVar(scip, linvars[v], &transvars[v]) );
               }
               assert(transvars[v] != NULL);
            }

            /* negate positive or negative variables for and-resultants*/
            for( v = 0; v < nandress; ++v )
            {
               if( mult * andvals[v] > 0.0 )
                  transvars[nlinvars + v] = andress[v];
               else
               {
                  SCIP_CALL( SCIPgetNegatedVar(scip, andress[v], &transvars[nlinvars + v]) );
               }
               assert(transvars[nlinvars + v] != NULL);
            }

            /* create the constraint */
            assert(!modifiable);
            SCIP_CALL( SCIPcreateConsSetpart(scip, &cons, name, nvars, transvars,
                  initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );

            created = TRUE;
            (*linconstype) = SCIP_SETPPC;

            /* release temporary memory */
            SCIPfreeBufferArray(scip, &transvars);
         }
         else if( (SCIPisInfinity(scip, -lhs) && SCIPisEQ(scip, rhs, 1.0 - ncoeffsnone))
            || (SCIPisEQ(scip, lhs, ncoeffspone - 1.0) && SCIPisInfinity(scip, rhs)) )
         {
            SCIPdebugMessage("linear pseudoboolean constraint will be a set packing constraint\n");

            /* check, if we have to multiply with -1 (negate the positive vars) or with +1 (negate the negative vars) */
            mult = SCIPisInfinity(scip, -lhs) ? +1 : -1;

            /* get temporary memory */
            SCIP_CALL( SCIPallocBufferArray(scip, &transvars, nvars) );

            /* negate positive or negative variables for linear variables */
            for( v = 0; v < nlinvars; ++v )
            {
               if( mult * linvals[v] > 0.0 )
                  transvars[v] = linvars[v];
               else
               {
                  SCIP_CALL( SCIPgetNegatedVar(scip, linvars[v], &transvars[v]) );
               }
               assert(transvars[v] != NULL);
            }

            /* negate positive or negative variables for and-resultants*/
            for( v = 0; v < nandress; ++v )
            {
               if( mult * andvals[v] > 0.0 )
                  transvars[nlinvars + v] = andress[v];
               else
               {
                  SCIP_CALL( SCIPgetNegatedVar(scip, andress[v], &transvars[nlinvars + v]) );
               }
               assert(transvars[nlinvars + v] != NULL);
            }

            /* create the constraint */
            assert(!modifiable);
            SCIP_CALL( SCIPcreateConsSetpack(scip, &cons, name, nvars, transvars,
                  initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );

            created = TRUE;
            (*linconstype) = SCIP_SETPPC;

            /* release temporary memory */
            SCIPfreeBufferArray(scip, &transvars);
         }
         else if( (SCIPisEQ(scip, lhs, 1.0 - ncoeffsnone) && SCIPisInfinity(scip, rhs))
            || (SCIPisInfinity(scip, -lhs) && SCIPisEQ(scip, rhs, ncoeffspone - 1.0)) )
         {
            SCIPwarningMessage("Does not expect this, because this constraint should be a logicor constraint.\n");
            SCIPdebugMessage("linear pseudoboolean constraint will be a set covering constraint\n");

            /* check, if we have to multiply with -1 (negate the positive vars) or with +1 (negate the negative vars) */
            mult = SCIPisInfinity(scip, rhs) ? +1 : -1;

            /* get temporary memory */
            SCIP_CALL( SCIPallocBufferArray(scip, &transvars, nvars) );

            /* negate positive or negative variables for linear variables */
            for( v = 0; v < nlinvars; ++v )
            {
               if( mult * linvals[v] > 0.0 )
                  transvars[v] = linvars[v];
               else
               {
                  SCIP_CALL( SCIPgetNegatedVar(scip, linvars[v], &transvars[v]) );
               }
               assert(transvars[v] != NULL);
            }

            /* negate positive or negative variables for and-resultants*/
            for( v = 0; v < nandress; ++v )
            {
               if( mult * andvals[v] > 0.0 )
                  transvars[nlinvars + v] = andress[v];
               else
               {
                  SCIP_CALL( SCIPgetNegatedVar(scip, andress[v], &transvars[nlinvars + v]) );
               }
               assert(transvars[nlinvars + v] != NULL);
            }

            /* create the constraint */
            assert(!modifiable);
            SCIP_CALL( SCIPcreateConsSetpack(scip, &cons, name, nvars, transvars,
                  initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );

            created = TRUE;
            (*linconstype) = SCIP_SETPPC;

            /* release temporary memory */
            SCIPfreeBufferArray(scip, &transvars);
         }
      }

      upgrconshdlr = SCIPfindConshdlr(scip, "knapsack");

      /* check, if linear constraint can be upgraded to a knapsack constraint
       * - all variables must be binary
       * - all coefficients must be integral
       * - exactly one of the sides must be infinite
       */
      if( upgrconshdlr != NULL && !created && (ncoeffspone + ncoeffsnone + ncoeffspint + ncoeffsnint == nvars) && (SCIPisInfinity(scip, -lhs) != SCIPisInfinity(scip, rhs)) )
      {
         SCIP_VAR** transvars;
         SCIP_Longint* weights;
         SCIP_Longint capacity;
         SCIP_Longint weight;
         int mult;

         SCIPdebugMessage("linear pseudoboolean constraint will be a knapsack constraint\n");
      
         /* get temporary memory */
         SCIP_CALL( SCIPallocBufferArray(scip, &transvars, nvars) );
         SCIP_CALL( SCIPallocBufferArray(scip, &weights, nvars) );

         /* if the right hand side is non-infinite, we have to negate all variables with negative coefficient;
          * otherwise, we have to negate all variables with positive coefficient and multiply the row with -1
          */
         if( SCIPisInfinity(scip, rhs) )
         {
            mult = -1;
            capacity = (SCIP_Longint)SCIPfeasFloor(scip, -lhs);
         }
         else
         {
            mult = +1;
            capacity = (SCIP_Longint)SCIPfeasFloor(scip, rhs);
         }

         /* negate positive or negative variables for linear variables */
         for( v = 0; v < nlinvars; ++v )
         {
            assert(SCIPisFeasIntegral(scip, linvals[v]));
            weight = mult * (SCIP_Longint)SCIPfeasFloor(scip, linvals[v]);
            if( weight > 0 )
            {
               transvars[v] = linvars[v];
               weights[v] = weight;
            }
            else
            {
               SCIP_CALL( SCIPgetNegatedVar(scip, linvars[v], &transvars[v]) );
               weights[v] = -weight;
               capacity -= weight;
            }
            assert(transvars[v] != NULL);
         }
         /* negate positive or negative variables for and-resultants */
         for( v = 0; v < nandress; ++v )
         {
            assert(SCIPisFeasIntegral(scip, andvals[v]));
            weight = mult * (SCIP_Longint)SCIPfeasFloor(scip, andvals[v]);
            if( weight > 0 )
            {
               transvars[nlinvars + v] = andress[v];
               weights[nlinvars + v] = weight;
            }
            else
            {
               SCIP_CALL( SCIPgetNegatedVar(scip, andress[v], &transvars[nlinvars + v]) );
               weights[nlinvars + v] = -weight;
               capacity -= weight;
            }
            assert(transvars[nlinvars + v] != NULL);
         }

         /* create the constraint */
         SCIP_CALL( SCIPcreateConsKnapsack(scip, &cons, name, nvars, transvars, weights, capacity,
               initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );

         created = TRUE;
         (*linconstype) = SCIP_KNAPSACK;

         /* free temporary memory */
         SCIPfreeBufferArray(scip, &weights);
         SCIPfreeBufferArray(scip, &transvars);
      }
#if 0

      upgrconshdlr = SCIPfindConshdlr(scip, "eqknapsack");

      /* check, if linear constraint can be upgraded to a knapsack constraint
       * - all variables must be binary
       * - all coefficients must be integral
       * - both sides must be infinite
       */
      if( upgrconshdlr != NULL && !created && (ncoeffspone + ncoeffsnone + ncoeffspint + ncoeffsnint == nvars) && SCIPisEQ(scip, lhs, rhs) )
      {
         SCIP_VAR** transvars;
         SCIP_Longint* weights;
         SCIP_Longint capacity;
         SCIP_Longint weight;
         int mult;

         assert(!SCIPisInfinity(scip, rhs));

         SCIPdebugMessage("linear pseudoboolean constraint will be a equality-knapsack constraint\n");
      
         /* get temporary memory */
         SCIP_CALL( SCIPallocBufferArray(scip, &transvars, nvars) );
         SCIP_CALL( SCIPallocBufferArray(scip, &weights, nvars) );

         if( SCIPisPositive(scip, rhs) )
         {
            mult = +1;
            capacity = (SCIP_Longint)SCIPfeasFloor(scip, rhs);
         }
         else
         {
            mult = -1;
            capacity = (SCIP_Longint)SCIPfeasFloor(scip, -rhs);
         }

         /* negate positive or negative variables for linear variables */
         for( v = 0; v < nlinvars; ++v )
         {
            assert(SCIPisFeasIntegral(scip, linvals[v]));
            weight = mult * (SCIP_Longint)SCIPfeasFloor(scip, linvals[v]);
            if( weight > 0 )
            {
               transvars[v] = linvars[v];
               weights[v] = weight;
            }
            else
            {
               SCIP_CALL( SCIPgetNegatedVar(scip, linvars[v], &transvars[v]) );
               weights[v] = -weight;
               capacity -= weight;
            }
            assert(transvars[v] != NULL);
         }
         /* negate positive or negative variables for and-resultants */
         for( v = 0; v < nandress; ++v )
         {
            assert(SCIPisFeasIntegral(scip, andvals[v]));
            weight = mult * (SCIP_Longint)SCIPfeasFloor(scip, andvals[v]);
            if( weight > 0 )
            {
               transvars[nlinvars + v] = andress[v];
               weights[nlinvars + v] = weight;
            }
            else
            {
               SCIP_CALL( SCIPgetNegatedVar(scip, andress[v], &transvars[nlinvars + v]) );
               weights[nlinvars + v] = -weight;
               capacity -= weight;
            }
            assert(transvars[nlinvars + v] != NULL);
         }

         /* create the constraint */
         SCIP_CALL( SCIPcreateConsEqKnapsack(scip, &cons, name, nvars, transvars, weights, capacity,
               initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );

         created = TRUE;
         (*linconstype) = SCIP_EQKNAPSACK;

         /* free temporary memory */
         SCIPfreeBufferArray(scip, &weights);
         SCIPfreeBufferArray(scip, &transvars);
      }
#endif
   }

   upgrconshdlr = SCIPfindConshdlr(scip, "linear");
   assert(created || upgrconshdlr != NULL);

   if( !created )
   {
      SCIP_CALL( SCIPcreateConsLinear(scip, &cons, name, nlinvars, linvars, linvals, lhs, rhs,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );
      
      (*linconstype) = SCIP_LINEAR;
   
      /* add all and-resultants */
      for( v = 0; v < nandress; ++v )
      {
         assert(andress[v] != NULL);

         /* add auxiliary variables to linear constraint */
         SCIP_CALL( SCIPaddCoefLinear(scip, cons, andress[v], andvals[v]) );
      }
   }
   
   assert(cons != NULL && *linconstype > SCIP_INVALIDCONS);

   SCIP_CALL( SCIPaddCons(scip, cons) );
   SCIPdebug( SCIP_CALL( SCIPprintCons(scip, cons, NULL) ) );

   *lincons = cons;
   SCIP_CALL( SCIPcaptureCons(scip, *lincons) );

   if( *linconstype == SCIP_LINEAR )
   {
      /* todo: make the constraint upgrade flag global, now it works only for the common linear constraint */
      /* mark linear constraint not to be upgraded - otherwise we loose control over it */
      SCIP_CALL( SCIPmarkDoNotUpgradeConsLinear(scip, cons) );
   }

   SCIP_CALL( SCIPreleaseCons(scip, &cons) );
   
   return SCIP_OKAY;
}

#if 0
/** checks pseudo boolean constraint for feasibility of given solution or current solution */
static
SCIP_RETCODE checkCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudo boolean constraint */
   SCIP_SOL*const        sol,                /**< solution to be checked, or NULL for current solution */
   SCIP_Bool*const       violated            /**< pointer to store whether the constraint is violated */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;
   SCIP_Real lhs;
   SCIP_Real rhs;
   SCIP_Real activity;
   int v;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(violated != NULL);

   SCIPdebugMessage("checking pseudo boolean constraint <%s>\n", SCIPconsGetName(cons));
   SCIPdebug( SCIP_CALL( SCIPprintCons(scip, cons, NULL) ) );
   
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* gets number of variables in linear constraint */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );

   /* allocate temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );

   /* get variables and coefficient of linear constraint */
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
   assert(nvars == 0 || (vars != NULL && coefs != NULL));

   /* get sides of linear constraint */
   SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &lhs, &rhs) );
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   *violated = FALSE;
   activity = 0.0;

   /* compute activity */
   for( v = nvars - 1; v >= 0; --v )
      activity += coefs[v] * SCIPgetSolVal(scip, sol, vars[v]);

   SCIPdebugMessage("  consdata activity=%.15g (lhs=%.15g, rhs=%.15g, sol=%p)\n",
      activity, consdata->lhs, consdata->rhs, (void*)sol);

   /* check linear part */
   if( SCIPisFeasLT(scip, activity, lhs) || SCIPisFeasGT(scip, activity, rhs) )
   {
      *violated = TRUE;
      SCIP_CALL( SCIPresetConsAge(scip, cons) );
   }
   else
   {
      SCIP_CALL( SCIPincConsAge(scip, cons) );
   }

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}
#endif

/** checks one original pseudoboolean constraint for feasibility of given solution */
static
SCIP_RETCODE checkOrigPbCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudo boolean constraint */
   SCIP_SOL*const        sol,                /**< solution to be checked, or NULL for current solution */
   SCIP_Bool*const       violated,           /**< pointer to store whether the constraint is violated */
   SCIP_Bool const       printreason         /**< should violation of constraint be printed */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;

   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;
   SCIP_Real lhs;
   SCIP_Real rhs;

   SCIP_VAR** linvars;
   SCIP_Real* lincoefs;
   int nlinvars;
   int v;

   SCIP_VAR** andress;
   SCIP_Real* andcoefs;
   int nandress;

   SCIP_CONS* andcons;
   SCIP_Real andvalue;
   SCIP_Real activity;
   int c;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(SCIPconsIsOriginal(cons));
   assert(violated != NULL);

   *violated = FALSE;

   SCIPdebugMessage("checking original pseudo boolean constraint <%s>\n", SCIPconsGetName(cons));
   SCIPdebug( SCIP_CALL( SCIPprintCons(scip, cons, NULL) ) );
   
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->lincons != NULL);
   assert(consdata->linconstype > SCIP_INVALIDCONS);
   assert(SCIPconsIsOriginal(consdata->lincons));
   
   /* gets number of variables in linear constraint */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );

   /* allocate temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andress, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nvars) );

   /* get sides of linear constraint */
   SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &lhs, &rhs) );
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   /* get variables and coefficient of linear constraint */
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
   assert(nvars == 0 || (vars != NULL && coefs != NULL));

   /* number of variables should be consistent, number of 'real' linear variables plus number of and-constraints should
    * have to be equal to the number of variables in the linear constraint
    */
   assert(consdata->nlinvars + consdata->nconsanddatas == nvars);

   nlinvars = 0;

   conshdlr = SCIPconsGetHdlr(cons);
   assert(conshdlr != NULL);
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->hashmap != NULL);
   
   nandress = 0;

   activity = 0.0;

   /* split variables into original and artificial variables and compute activity on normal linear variables(without
    * terms)
    */
   for( v = 0; v < nvars; ++v )
   { 
      assert(vars[v] != NULL);

      if( !SCIPhashmapExists(conshdlrdata->hashmap, (void*)(vars[v])) ) // ????????? strstr(SCIPvarGetName(vars[v]), ARTIFICIALVARNAMEPREFIX) == NULL )
      {
         activity += coefs[v] * SCIPgetSolVal(scip, sol, vars[v]);

         linvars[nlinvars] = vars[v];
         lincoefs[nlinvars] = coefs[v];
         ++nlinvars;
      }
      else
      {
         andress[nandress] = vars[v];
         andcoefs[nandress] = coefs[v];
         ++nandress;
      }
   }
   assert(nandress == consdata->nconsanddatas);

   SCIPdebugMessage("nlinvars = %d, nandress = %d\n", nlinvars, nandress);
   SCIPdebugMessage("linear activity = %g\n", activity);

   /* compute and add solution values on terms */
   for( c = consdata->nconsanddatas - 1; c >= 0; --c )
   {
      SCIP_VAR** andvars;
      int nandvars;
      SCIP_VAR* res;

      andcons = consdata->consanddatas[c]->origcons;
      assert(andcons != NULL);
      
      andvars = SCIPgetVarsAnd(scip, andcons);
      nandvars = SCIPgetNVarsAnd(scip, andcons);
      res = SCIPgetResultantAnd(scip, andcons);
      assert(nandvars == 0 || (andvars != NULL && res != NULL));
      assert(res == andress[c]);

      andvalue = 1;
      /* check if the and-constraint is violated */
      for( v = nandvars - 1; v >= 0; --v )
      {
         andvalue *= SCIPgetSolVal(scip, sol, andvars[v]);
         if( SCIPisFeasZero(scip, andvalue) )
            break;
      }
      activity += andvalue * andcoefs[c];
   }
   SCIPdebugMessage("lhs = %g, overall activity = %g, rhs = %g\n", lhs, activity, rhs);

   /* check left hand side for violation */
   if( SCIPisFeasLT(scip, activity, lhs) )
   {
      if( printreason )
      {
         SCIP_CALL( SCIPprintCons(scip, cons, NULL ) );
         SCIPinfoMessage(scip, NULL, "violation: left hand side is violated by %.15g\n", lhs - activity);
      }

      *violated = TRUE;
   }

   /* check right hand side for violation */
   if( SCIPisFeasGT(scip, activity, rhs) )
   {
      if( printreason )
      {
         SCIP_CALL( SCIPprintCons(scip, cons, NULL ) );
         SCIPinfoMessage(scip, NULL, "violation: right hand side is violated by %.15g\n", activity - rhs);
      }

      *violated = TRUE;
   }

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &andcoefs);
   SCIPfreeBufferArray(scip, &andress);
   SCIPfreeBufferArray(scip, &lincoefs);
   SCIPfreeBufferArray(scip, &linvars);
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/** checks all and-constraints inside the pseudoboolean constraint handler for feasibility of given solution or current solution */
static
SCIP_RETCODE checkAndConss(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*const   conshdlr,           /**< pseudo boolean constraint handler */
   SCIP_SOL*const        sol,                /**< solution to be checked, or NULL for current solution */
   SCIP_Bool*const       violated            /**< pointer to store whether the constraint is violated */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONS* andcons;
   SCIP_VAR** vars;
   SCIP_VAR* res;
   int nvars;
   SCIP_Real andvalue;
   int c;
   int v;
   
   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(violated != NULL);
   
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   
   *violated = FALSE;

   for( c = conshdlrdata->nallconsanddatas - 1; c >= 0; --c )
   {
      if( conshdlrdata->allconsanddatas[c]->deleted )
         continue;

      andcons = conshdlrdata->allconsanddatas[c]->cons;
      
      if( andcons == NULL || !SCIPconsIsActive(andcons) )
         continue;
      
      vars = SCIPgetVarsAnd(scip, andcons);
      nvars = SCIPgetNVarsAnd(scip, andcons);
      res = SCIPgetResultantAnd(scip, andcons);
      assert(nvars == 0 || (vars != NULL && res != NULL));

      andvalue = 1;
      /* check if the and-constraint is violated */
      for( v = nvars - 1; v >= 0; --v )
      {
         andvalue *= SCIPgetSolVal(scip, sol, vars[v]);
         if( SCIPisFeasZero(scip, andvalue) )
            break;
      }

      /* check for violation and update aging */
      if( !SCIPisFeasEQ(scip, andvalue, SCIPgetSolVal(scip, sol, res)) )
      {
         SCIP_CALL( SCIPresetConsAge(scip, andcons) );

         *violated = TRUE;
         break;
      }
      else
      {
         SCIP_CALL( SCIPincConsAge(scip, andcons) );
      }
   }

   return SCIP_OKAY;
}

/** creates by copying and captures a linear constraint */
static
SCIP_RETCODE copyConsPseudoboolean(
   SCIP*const            targetscip,         /**< target SCIP data structure */
   SCIP_CONS**           targetcons,         /**< pointer to store the created target constraint */
   SCIP*const            sourcescip,         /**< source SCIP data structure */
   SCIP_CONS*const       sourcecons,         /**< source constraint which will be copied */
   const char*           name,               /**< name of constraint */
   SCIP_HASHMAP*const    varmap,             /**< a SCIP_HASHMAP mapping variables of the source SCIP to corresponding
                                              *   variables of the target SCIP */
   SCIP_HASHMAP*const    consmap,            /**< a hashmap to store the mapping of source constraints to the corresponding
                                              *   target constraints */
   SCIP_Bool const       initial,            /**< should the LP relaxation of constraint be in the initial LP? */
   SCIP_Bool const       separate,           /**< should the constraint be separated during LP processing? */
   SCIP_Bool const       enforce,            /**< should the constraint be enforced during node processing? */
   SCIP_Bool const       check,              /**< should the constraint be checked for feasibility? */
   SCIP_Bool const       propagate,          /**< should the constraint be propagated during node processing? */
   SCIP_Bool const       local,              /**< is constraint only valid locally? */
   SCIP_Bool const       modifiable,         /**< is constraint modifiable (subject to column generation)? */
   SCIP_Bool const       dynamic,            /**< is constraint subject to aging? */
   SCIP_Bool const       removable,          /**< should the relaxation be removed from the LP due to aging or cleanup? */
   SCIP_Bool const       stickingatnode,     /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node? */
   SCIP_Bool const       global,             /**< create a global or a local copy? */
   SCIP_Bool*const       valid               /**< pointer to store if the copying was valid */
   )
{
   SCIP_CONSDATA* sourceconsdata;
   SCIP_CONS* sourcelincons;
   
   assert(targetscip != NULL);
   assert(targetcons != NULL);
   assert(sourcescip != NULL);
   assert(sourcecons != NULL);
   assert(strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(sourcecons)), CONSHDLR_NAME) == 0);
   assert(valid != NULL);

   *valid = TRUE;

   sourceconsdata = SCIPconsGetData(sourcecons);
   assert(sourceconsdata != NULL);

   /* get linear constraint */
   sourcelincons = sourceconsdata->lincons;
   assert(sourcelincons != NULL);

   /* get copied version of linear constraint */
   if( !SCIPconsIsDeleted(sourcelincons) )
   {
      SCIP_CONSHDLR* conshdlrlinear;
      SCIP_CONS* targetlincons;
      SCIP_CONS** targetandconss;
      SCIP_Real* targetandcoefs;
      int ntargetandconss;
      SCIP_LINEARCONSTYPE targetlinconstype;

      targetlinconstype = sourceconsdata->linconstype;
     
      switch( targetlinconstype )
      {
      case SCIP_LINEAR:
         conshdlrlinear = SCIPfindConshdlr(sourcescip, "linear");
         assert(conshdlrlinear != NULL);
         break;
      case SCIP_LOGICOR:
         conshdlrlinear = SCIPfindConshdlr(sourcescip, "logicor");
         assert(conshdlrlinear != NULL);
         break;
      case SCIP_KNAPSACK:
         conshdlrlinear = SCIPfindConshdlr(sourcescip, "knapsack");
         assert(conshdlrlinear != NULL);
         break;
      case SCIP_SETPPC:
         conshdlrlinear = SCIPfindConshdlr(sourcescip, "setppc");
         assert(conshdlrlinear != NULL);
         break;
#if 0
      case SCIP_EQKNAPSACK:
         conshdlrlinear = SCIPfindConshdlr(sourcescip, "eqknapsack");
         assert(conshdlrlinear != NULL);
         break;
#endif
      default:
         SCIPerrorMessage("unknown linear constraint type\n");
         return SCIP_INVALIDDATA;
      }

      if( conshdlrlinear == NULL )
      {
         SCIPerrorMessage("linear constraint handler not found\n");
         return SCIP_INVALIDDATA;
      }

      /* copy linear constraint */
      SCIP_CALL( SCIPgetConsCopy(sourcescip, targetscip, sourcelincons, &targetlincons, conshdlrlinear, varmap, consmap, SCIPconsGetName(sourcelincons),
            SCIPconsIsInitial(sourcelincons), SCIPconsIsSeparated(sourcelincons), SCIPconsIsEnforced(sourcelincons), SCIPconsIsChecked(sourcelincons),
            SCIPconsIsPropagated(sourcelincons), SCIPconsIsLocal(sourcelincons), SCIPconsIsModifiable(sourcelincons), SCIPconsIsDynamic(sourcelincons),
            SCIPconsIsRemovable(sourcelincons), SCIPconsIsStickingAtNode(sourcelincons), global, valid) );
      
      if( *valid )
      {
         assert(targetlincons != NULL);
         assert(SCIPconsGetHdlr(targetlincons) != NULL);
         /* @NOTE: due to copying special linear constraints, now leads only to simple linear constraints, we check that
          *        our target constraint handler is the same as our source constraint handler of the linear constraint,
          *        if not copying was not valid
          */
#if 0
         *valid &= (strcmp(SCIPconshdlrGetName(conshdlrlinear), SCIPconshdlrGetName(SCIPconsGetHdlr(targetlincons)) ) == 0 );
#else
         if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(targetlincons)), "linear") == 0 )
            targetlinconstype = SCIP_LINEAR;
#endif
      }

      targetandconss = NULL;
      targetandcoefs = NULL;
      ntargetandconss = 0;
      
      if( *valid )
      {        
         SCIP_CONSHDLR* conshdlrand;
         SCIP_CONS* oldcons;
         SCIP_Bool validand;
         int c;
         int nsourceandconss;

         conshdlrand = SCIPfindConshdlr(sourcescip, "and");
         assert(conshdlrand != NULL);

         nsourceandconss = sourceconsdata->nconsanddatas;
         
         /* allocate temporary memory */
         SCIP_CALL( SCIPallocBufferArray(sourcescip, &targetandconss, nsourceandconss) );
         SCIP_CALL( SCIPallocBufferArray(sourcescip, &targetandcoefs, nsourceandconss) );

         for( c = 0 ; c < nsourceandconss; ++c )
         {
            CONSANDDATA* consanddata;
            
            consanddata = sourceconsdata->consanddatas[c];
            assert(consanddata != NULL);

            oldcons = consanddata->cons;
            assert(oldcons != NULL);

            validand = TRUE;

            /* copy and-constraints */
            SCIP_CALL( SCIPgetConsCopy(sourcescip, targetscip, oldcons, &targetandconss[ntargetandconss], conshdlrand, varmap, consmap, SCIPconsGetName(oldcons),
                  SCIPconsIsInitial(oldcons), SCIPconsIsSeparated(oldcons), SCIPconsIsEnforced(oldcons), SCIPconsIsChecked(oldcons),
                  SCIPconsIsPropagated(oldcons), SCIPconsIsLocal(oldcons), SCIPconsIsModifiable(oldcons), SCIPconsIsDynamic(oldcons),
                  SCIPconsIsRemovable(oldcons), SCIPconsIsStickingAtNode(oldcons), global, &validand) );

            *valid &= validand;

            if( validand )
            {
               targetandcoefs[ntargetandconss] = sourceconsdata->andcoefs[c];
               ++ntargetandconss;
            }
         }
      }

      /* no correct pseudoboolean constraint */
      if( ntargetandconss == 0 )
      {
         SCIPdebugMessage("no and-constraints copied for pseudoboolean constraint <%s>\n", SCIPconsGetName(sourcecons));
         *valid = FALSE;
      }

      if( *valid )
      {
         SCIP_VAR* intvar;
         SCIP_VAR* indvar;
         const char* consname;

         /* third the indicator and artificial integer variable part */
         assert(sourceconsdata->issoftcons == (sourceconsdata->indvar != NULL));
         indvar = sourceconsdata->indvar;
         intvar = sourceconsdata->intvar;
         
         /* copy indicator variable */
         if( indvar != NULL )
         {
            assert(*valid);
            SCIP_CALL( SCIPgetVarCopy(sourcescip, targetscip, indvar, &indvar, varmap, consmap, global, valid) );
            assert(!(*valid) || indvar != NULL);
         }
         /* copy artificial integer variable */
         if( intvar != NULL && *valid )
         {
            SCIP_CALL( SCIPgetVarCopy(sourcescip, targetscip, intvar, &intvar, varmap, consmap, global, valid) );
            assert(!(*valid) || intvar != NULL);
         }
         
         if( name != NULL )
            consname = name;
         else
            consname = SCIPconsGetName(sourcecons);
         
         /* create new pseudoboolean constraint */
         SCIP_CALL( SCIPcreateConsPseudobooleanWithConss(targetscip, targetcons, consname, 
               targetlincons, targetlinconstype, targetandconss, targetandcoefs, ntargetandconss,
               indvar, sourceconsdata->weight, sourceconsdata->issoftcons, intvar, 
               sourceconsdata->lhs, sourceconsdata->rhs, 
               initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode) );
      }
      else
      {
         SCIPverbMessage(sourcescip, SCIP_VERBLEVEL_MINIMAL, NULL, "could not copy constraint <%s>\n", SCIPconsGetName(sourcecons));
      }

      /* free temporary memory */
      SCIPfreeBufferArrayNull(sourcescip, &targetandcoefs);
      SCIPfreeBufferArrayNull(sourcescip, &targetandconss);
   }
   else
      *valid = FALSE;

   return SCIP_OKAY;
}

/* compute all changes in consanddatas array */
static
SCIP_RETCODE computeConsAndDataChanges(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLRDATA*const conshdlrdata      /**< pseudoboolean constraint handler data */
   )
{
   CONSANDDATA** allconsanddatas;
   CONSANDDATA* consanddata;
   int c;

   assert(scip != NULL);
   assert(conshdlrdata != NULL);

   allconsanddatas = conshdlrdata->allconsanddatas;
   assert(allconsanddatas != NULL);
   assert(conshdlrdata->nallconsanddatas > 0);
   assert(conshdlrdata->nallconsanddatas <= conshdlrdata->sallconsanddatas);

   for( c = conshdlrdata->nallconsanddatas - 1; c >= 0; --c )
   {
      SCIP_CONS* cons;
      SCIP_VAR** vars;
      int nvars;
      SCIP_VAR** newvars;
      int nnewvars;
      int v;
      
      consanddata = allconsanddatas[c];

      if( consanddata->deleted )
         continue;

      vars = consanddata->vars;
      nvars = consanddata->nvars;
      assert(nvars == 0 || vars != NULL);
      assert(consanddata->nnewvars == 0 && ((consanddata->snewvars > 0) == (consanddata->newvars != NULL)));

      if( nvars == 0 )
      {
#ifndef NDEBUG
         /* if an old consanddata-object has no variables left there should be no new variables */
         if( consanddata->cons != NULL )
         {
            nnewvars = SCIPgetNVarsAnd(scip, consanddata->cons);
            assert(nnewvars == 0);
         }
#endif
         continue;
      }

      cons = consanddata->cons;
      assert(cons != NULL);

      if( SCIPconsIsDeleted(cons) )
      {
         continue;
      }
      
      /* sort and-variables */
      if( !SCIPisAndConsSorted(scip, consanddata->cons) )
      {
         SCIP_CALL( SCIPsortAndCons(scip, consanddata->cons) );
         assert(SCIPisAndConsSorted(scip, consanddata->cons));
      }
      
      /* get new and-variables */
      nnewvars = SCIPgetNVarsAnd(scip, consanddata->cons);
      newvars = SCIPgetVarsAnd(scip, consanddata->cons);
   
#ifndef NDEBUG
      /* check that old variables are sorted */
      for( v = nvars - 1; v > 0; --v )
         assert(SCIPvarGetIndex(vars[v]) > SCIPvarGetIndex(vars[v - 1]));
      /* check that new variables are sorted */
      for( v = nnewvars - 1; v > 0; --v )
         assert(SCIPvarGetIndex(newvars[v]) > SCIPvarGetIndex(newvars[v - 1]));
#endif

      /* check for changings, if and-constraint did not change we do not need to copy all variables */
      if( nvars == nnewvars )
      {
         SCIP_Bool changed;
         
         changed = FALSE;
         
         /* check each variable */
         for( v = nvars - 1; v >= 0; --v )
         {
            if( vars[v] != newvars[v] )
            {
               changed = TRUE;
               break;
            }
         }
         
         if( !changed )
            continue;
      }
      
      /* resize newvars array if necessary */
      if( nnewvars > consanddata->snewvars )
      {
         SCIP_CALL( SCIPensureBlockMemoryArray(scip, &(consanddata->newvars), &(consanddata->snewvars), nnewvars) );
      }
      
      /* copy all variables */
      BMScopyMemoryArray(consanddata->newvars, newvars, nnewvars);
      consanddata->nnewvars = nnewvars;

      /* capture all variables */
      for( v = consanddata->nnewvars - 1; v >= 0; --v )
      {
         /* in original problem the variables was already deleted */
         assert(consanddata->newvars[v] != NULL);
         SCIP_CALL( SCIPcaptureVar(scip, consanddata->newvars[v]) );
      }
   }

   return SCIP_OKAY;
}

/* remove old locks */
static
SCIP_RETCODE removeOldLocks(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   CONSANDDATA*const     consanddata,        /**< CONSANDDATA object for which we want to delete the locks and the
                                              *   capture of the corresponding and-constraint */
   SCIP_Real const       coef,               /**< coefficient which led to old locks */
   SCIP_Real const       lhs,                /**< left hand side which led to old locks */
   SCIP_Real const       rhs                /**< right hand side which led to old locks */
   )
{
   assert(scip != NULL);
   assert(cons != NULL);
   assert(consanddata != NULL);
   assert(!SCIPisInfinity(scip, coef) && !SCIPisInfinity(scip, -coef));
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   /* remove rounding locks */
   SCIP_CALL( unlockRoundingAndCons(scip, cons, consanddata, coef, lhs, rhs) );

   assert(consanddata->cons != NULL);

   return SCIP_OKAY;
}

/* add new locks */
static
SCIP_RETCODE addNewLocks(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   CONSANDDATA*const     consanddata,        /**< CONSANDDATA object for which we want to delete the locks and the
                                              *   capture of the corresponding and-constraint */
   SCIP_Real const       coef,               /**< coefficient which lead to new locks */
   SCIP_Real const       lhs,                /**< left hand side which lead to new locks */
   SCIP_Real const       rhs                 /**< right hand side which lead to new locks */
   )
{
   assert(scip != NULL);
   assert(cons != NULL);
   assert(consanddata != NULL);
   assert(!SCIPisInfinity(scip, coef) && !SCIPisInfinity(scip, -coef));
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   /* add rounding locks due to old variables in consanddata object */
   SCIP_CALL( lockRoundingAndCons(scip, cons, consanddata, coef, lhs, rhs) );

   assert(consanddata->cons != NULL);

   return SCIP_OKAY;
}

/* update all locks inside this constraint and all captures on all and-constraints */
static
SCIP_RETCODE correctLocksAndCaptures(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_CONSHDLRDATA*const conshdlrdata,     /**< pseudoboolean constraint handler data */
   SCIP_Real const       newlhs,             /**< new left hand side of pseudoboolean constraint */
   SCIP_Real const       newrhs,             /**< new right hand side of pseudoboolean constraint */
   SCIP_VAR**const       andress,            /**< current and-resultants in pseudoboolean constraint */
   SCIP_Real*const       andcoefs,           /**< current and-resultants-coeffcients in pseudoboolean constraint */
   int const             nandress            /**< number of current and-resultants in pseudoboolean constraint */
   )
{
   CONSANDDATA** newconsanddatas;
   int nnewconsanddatas;
   int snewconsanddatas;
   SCIP_Real* newandcoefs;
   SCIP_Real* oldandcoefs;
   CONSANDDATA** consanddatas;
   int nconsanddatas;
   SCIP_CONSDATA* consdata;
   int c;
   int c1;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->hashmap != NULL);
   assert(nandress == 0 || (andress != NULL && andcoefs != NULL));
   assert(!SCIPisInfinity(scip, newlhs));
   assert(!SCIPisInfinity(scip, -newrhs));
   assert(SCIPisLE(scip, newlhs, newrhs));

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   consanddatas = consdata->consanddatas;
   oldandcoefs = consdata->andcoefs;
   nconsanddatas = consdata->nconsanddatas;
   assert(nconsanddatas == 0 || (consanddatas != NULL && oldandcoefs != NULL));

#ifndef NDEBUG
   /* check that and-resultants are sorted, and coefficents are not zero */
   for( c = nandress - 1; c > 0; --c )
   {
      assert(!SCIPisZero(scip, andcoefs[c]));
      assert(SCIPvarGetIndex(andress[c]) > SCIPvarGetIndex(andress[c - 1]));
   }
   /* check that consanddata objects are sorted due to the index of the corresponding resultants, and coefficents are
    * not zero 
    */
   for( c = nconsanddatas - 1; c > 0; --c )
   {
      SCIP_VAR* res1;
      SCIP_VAR* res2;

      assert(consanddatas[c] != NULL);

      if( consanddatas[c]->deleted )
         continue;

      assert(!SCIPisZero(scip, oldandcoefs[c]));
      assert(consanddatas[c - 1] != NULL);

      if( consanddatas[c - 1]->deleted )
         continue;

      assert(!SCIPisZero(scip, oldandcoefs[c - 1]));

      assert(consanddatas[c]->cons != NULL);
      res1 = SCIPgetResultantAnd(scip, consanddatas[c]->cons);
      assert(res1 != NULL);
      assert(consanddatas[c - 1]->cons != NULL);
      res2 = SCIPgetResultantAnd(scip, consanddatas[c - 1]->cons);
      assert(res2 != NULL);

      assert(SCIPvarGetIndex(res1) > SCIPvarGetIndex(res2));
   }
#endif

   snewconsanddatas = nconsanddatas + nandress;

   /* allocate new block memory arrays */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &newconsanddatas, snewconsanddatas) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &newandcoefs, snewconsanddatas) );

   nnewconsanddatas = 0;

   /* collect new consanddata objects and update locks and captures */
   for( c = 0, c1 = 0; c < nconsanddatas && c1 < nandress; )
   {
      SCIP_CONS* andcons;
      SCIP_VAR* res1;
      SCIP_VAR* res2;

      assert(consanddatas[c] != NULL);

      /* consanddata object could have been deleted in the last presolving round */ 
      if( consanddatas[c]->deleted )
      {
         ++c;
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
         continue;
      }

      andcons = consanddatas[c]->cons;
      assert(andcons != NULL);
#if 1
      if( andcons == NULL )
      {
         ++c;
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
         continue;
      }
#else
      if( andcons == NULL )
      {
         /* remove old locks */
         SCIP_CALL( removeOldLocks(scip, cons, consanddatas[c], oldandcoefs[c], consdata->lhs, consdata->rhs) );
         ++c;
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
         continue;
      }
      else if( SCIPconsIsDeleted(andcons) )
      {
         /* remove rounding locks, because this data will only be  */
         SCIP_CALL( unlockRoundingAndCons(scip, cons, consanddatas[c], oldandcoefs[c], consdata->lhs, consdata->rhs) );
         ++c;
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
         continue;
      }
#endif
      assert(andcons != NULL);

      /* get and-resultants of consanddata object in constraint data */
      res1 = SCIPgetResultantAnd(scip, andcons);
      assert(res1 != NULL);
      assert(SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res1) == consanddatas[c]);

      /* get and-resultants in new corresponding linear constraint */
      res2 = andress[c1];
      assert(res2 != NULL);
      assert(SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res2) != NULL);

      /* collect new consanddata objects in sorted order due to the variable index of corresponding and-resultants */
      if( SCIPvarGetIndex(res1) < SCIPvarGetIndex(res2) )
      {
         /* remove old locks */
         SCIP_CALL( removeOldLocks(scip, cons, consanddatas[c], oldandcoefs[c], consdata->lhs, consdata->rhs) );

#if 0         
         assert(consanddatas[c]->nuses > 0);
         --(consanddatas[c]->nuses);
#endif

         ++c;
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
      }
      else if( SCIPvarGetIndex(res1) > SCIPvarGetIndex(res2) )
      {
         newconsanddatas[nnewconsanddatas] = (CONSANDDATA*) SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res2);
         newandcoefs[nnewconsanddatas] = andcoefs[c1];
      
         /* add new locks */
         SCIP_CALL( addNewLocks(scip, cons, newconsanddatas[nnewconsanddatas], newandcoefs[nnewconsanddatas], newlhs, newrhs) );
#if 0         
         assert(newconsanddatas[nnewconsanddatas]->nuses > 0);
         --(newconsanddatas[nnewconsanddatas]->nuses);
#endif
         ++c1;
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
         ++nnewconsanddatas;
      }
      else
      {
         SCIP_Bool coefsignchanged;
         SCIP_Bool lhschanged;
         SCIP_Bool rhschanged;

         assert(SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res2) == consanddatas[c]);
         
         /* copy old consanddata object and new coefficent */
         newconsanddatas[nnewconsanddatas] = consanddatas[c];

         if( !SCIPisEQ(scip, oldandcoefs[c], andcoefs[c1]) )
            consdata->upgradetried = FALSE;

         newandcoefs[nnewconsanddatas] = andcoefs[c1];

         coefsignchanged = (oldandcoefs[c] < 0 && andcoefs[c1] > 0) || (oldandcoefs[c] > 0 && andcoefs[c1] < 0);
         lhschanged = (SCIPisInfinity(scip, -consdata->lhs) && !SCIPisInfinity(scip, -newlhs)) || (!SCIPisInfinity(scip, -consdata->lhs) && SCIPisInfinity(scip, -newlhs))
            || (consdata->lhs < 0 && newlhs > 0) || (consdata->lhs > 0 && newlhs < 0);
         rhschanged = (SCIPisInfinity(scip, consdata->rhs) && !SCIPisInfinity(scip, newrhs)) || (!SCIPisInfinity(scip, consdata->rhs) && SCIPisInfinity(scip, newrhs))
            || (consdata->rhs < 0 && newrhs > 0) || (consdata->rhs > 0 && newrhs < 0);

         /* update or renew locks */
         if( !coefsignchanged && !lhschanged && !rhschanged )
         {
            if( newconsanddatas[nnewconsanddatas]->nnewvars > 0 )
            {
               /* update locks */
               SCIP_CALL( removeOldLocks(scip, cons, newconsanddatas[nnewconsanddatas], oldandcoefs[c], consdata->lhs, consdata->rhs) );
               SCIP_CALL( addNewLocks(scip, cons, newconsanddatas[nnewconsanddatas], newandcoefs[nnewconsanddatas], newlhs, newrhs) );
               consdata->changed = TRUE;
               consdata->upgradetried = FALSE;
            }
         }
         else
         {
            /* renew locks */
            SCIP_CALL( removeOldLocks(scip, cons, newconsanddatas[nnewconsanddatas], oldandcoefs[c], consdata->lhs, consdata->rhs) );
            SCIP_CALL( addNewLocks(scip, cons, newconsanddatas[nnewconsanddatas], newandcoefs[nnewconsanddatas], newlhs, newrhs) );
            consdata->changed = TRUE;
            consdata->upgradetried = FALSE;
         }

         ++c;
         ++c1;
         ++nnewconsanddatas;
      }
   }      
   
   /* add all remaining consanddatas and update locks and captures */
   if( c < nconsanddatas )
   {
      assert(c1 == nandress);
      
      for( ; c < nconsanddatas; ++c )
      {
         SCIP_CONS* andcons;
#ifndef NDEBUG
         SCIP_VAR* res1;

         assert(consanddatas[c] != NULL);
         
         andcons = consanddatas[c]->cons;
         if( andcons != NULL )
         {
            res1 = SCIPgetResultantAnd(scip, andcons);
            assert(res1 != NULL);
            assert(SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res1) == consanddatas[c]);
         }
#endif
         if( andcons == NULL )
         {
            consdata->changed = TRUE;
            consdata->upgradetried = FALSE;
            continue;
         }

         /* remove old locks */
         SCIP_CALL( removeOldLocks(scip, cons, consanddatas[c], oldandcoefs[c], consdata->lhs, consdata->rhs) );
#if 0         
         assert(consanddatas[c]->nuses > 0);
         --(consanddatas[c]->nuses);
#endif
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
      }
   }
   else if( c1 < nandress )
   {
      for( ; c1 < nandress; ++c1 )
      {
         SCIP_VAR* res2;

         res2 = andress[c1];
         assert(res2 != NULL);
         assert(SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res2) != NULL);

         newconsanddatas[nnewconsanddatas] = (CONSANDDATA*) SCIPhashmapGetImage(conshdlrdata->hashmap, (void*)res2);

         /* add new locks */
         SCIP_CALL( addNewLocks(scip, cons, newconsanddatas[nnewconsanddatas], newandcoefs[nnewconsanddatas], newlhs, newrhs) );
#if 0         
         assert(newconsanddatas[nnewconsanddatas]->nuses > 0);
         --(newconsanddatas[nnewconsanddatas]->nuses);
#endif

         ++nnewconsanddatas;
         consdata->changed = TRUE;
         consdata->upgradetried = FALSE;
      }
   }
   assert(c == nconsanddatas && c1 == nandress);

   /* delete old and-coefficients and consanddata objects */
   SCIPfreeBlockMemoryArray(scip, &(consdata->andcoefs), consdata->sconsanddatas);
   SCIPfreeBlockMemoryArray(scip, &(consdata->consanddatas), consdata->sconsanddatas);

   if( !SCIPisEQ(scip, consdata->lhs, newlhs) || !SCIPisEQ(scip, consdata->rhs, newrhs) )
   {
      consdata->upgradetried = FALSE;
      consdata->lhs = newlhs;
      consdata->rhs = newrhs;
   }

   consdata->consanddatas = newconsanddatas;
   consdata->andcoefs = newandcoefs;
   consdata->nconsanddatas = nnewconsanddatas;
   consdata->sconsanddatas = snewconsanddatas;

   /* update number of linear variables without and-resultants */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &(consdata->nlinvars)) );
   consdata->nlinvars -= nnewconsanddatas;
   
#ifndef NDEBUG
   consanddatas = consdata->consanddatas;
   nconsanddatas = consdata->nconsanddatas;
   assert(nconsanddatas == 0 || consanddatas != NULL);

   /* check that consanddata objects are sorted due to the index of the corresponding resultants */
   for( c = nconsanddatas - 1; c > 0; --c )
   {
      SCIP_VAR* res1;
      SCIP_VAR* res2;

      assert(consanddatas[c] != NULL);
      assert(consanddatas[c]->cons != NULL);
      res1 = SCIPgetResultantAnd(scip, consanddatas[c]->cons);
      assert(res1 != NULL);
      assert(consanddatas[c - 1] != NULL);
      assert(consanddatas[c - 1]->cons != NULL);
      res2 = SCIPgetResultantAnd(scip, consanddatas[c - 1]->cons);
      assert(res2 != NULL);

      assert(SCIPvarGetIndex(res1) > SCIPvarGetIndex(res2));
   }
#endif

   return SCIP_OKAY;
}

/** adds cliques of the pseudoboolean constraint to the global clique table */
static
SCIP_RETCODE addCliques(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_Bool*const       cutoff,             /**< pointer to store whether the node can be cut off */
   int*const             naggrvars,          /**< pointer to count the number of aggregated variables */
   int*const             nchgbds             /**< pointer to count the number of performed bound changes */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;
   SCIP_VAR** linvars;
   SCIP_Real* lincoefs;
   int nlinvars;
   SCIP_VAR** andress;
   SCIP_Real* andcoefs;
   int nandress;
   int c;
   int v2;
   int v1;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(cutoff != NULL);
   assert(naggrvars != NULL);
   assert(nchgbds != NULL);
   assert(SCIPconsIsActive(cons));

   *cutoff = FALSE;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   /* if we have no and-constraints left, we should not be here and this constraint should be deleted (only the linaer should survive) */
   assert(consdata->nconsanddatas > 0);
   
   /* check whether the cliques have already been added */
   if( consdata->cliquesadded )
      return SCIP_OKAY;

   consdata->cliquesadded = TRUE;

   /* check standard pointers and sizes */
   assert(consdata->lincons != NULL);
   assert(SCIPconsIsActive(consdata->lincons));
   assert(consdata->linconstype > SCIP_INVALIDCONS);
   assert(consdata->consanddatas != NULL);
   assert(consdata->nconsanddatas > 0);
   assert(consdata->nconsanddatas <= consdata->sconsanddatas);

   /* check number of linear variables */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );
   assert(nvars == consdata->nlinvars + consdata->nconsanddatas);

   /* get temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andress, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nvars) );

   /* get variables and coefficients */ 
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
   assert(nvars == 0 || (vars != NULL && coefs != NULL));

   /* calculate all not artificial linear variables and all artificial and-resultants which will be ordered like the
    * 'consanddatas' such that the and-resultant of the and-constraint is the and-resultant in the 'andress' array
    * afterwards 
    */
   SCIP_CALL( getLinVarsAndAndRess(scip, cons, vars, coefs, nvars, linvars, lincoefs, &nlinvars, andress, andcoefs, &nandress) );

   assert(nandress == consdata->nconsanddatas);
   assert(consdata->consanddatas != NULL);

   /* find cliques from linear variable to and-resultant */
   for( c = nandress - 1; c >= 0; --c )
   {
      CONSANDDATA* consanddata;
      SCIP_VAR** andvars;
      int nandvars;
      
      consanddata = consdata->consanddatas[c];
      assert(consanddata != NULL);
      
      assert(SCIPgetResultantAnd(scip, consanddata->cons) == andress[c]);

      /* choose correct variable array */
      if( consanddata->nnewvars > 0 )
      {
         andvars = consanddata->newvars;
         nandvars = consanddata->nnewvars;
      }
      else
      {
         andvars = consanddata->vars;
         nandvars = consanddata->nvars;
      }

      for( v1 = nandvars - 1; v1 >= 0; --v1 )
      {
         SCIP_VAR* var1;
         SCIP_Bool values[2];

         var1 = andvars[v1];
         if( !SCIPvarIsActive(var1) && (!SCIPvarIsNegated(var1) || !SCIPvarIsActive(SCIPvarGetNegationVar(var1))) )
            continue;

         /* get active counterpart to check for common cliques */
         if( SCIPvarGetStatus(var1) == SCIP_VARSTATUS_NEGATED )
         {
            var1 = SCIPvarGetNegationVar(var1);
            values[0] = FALSE;
         }
         else
            values[0] = TRUE;

         for( v2 = nlinvars - 1; v2 >= 0; --v2 )
         {
            SCIP_VAR* var2;

            var2 = linvars[v2];
            if( !SCIPvarIsActive(var2) && (!SCIPvarIsNegated(var2) || !SCIPvarIsActive(SCIPvarGetNegationVar(var2))) )
               continue;

            /* get active counterpart to check for common cliques */
            if( SCIPvarGetStatus(var2) == SCIP_VARSTATUS_NEGATED )
            {
               var2 = SCIPvarGetNegationVar(var2);
               values[1] = FALSE;
            }
            else
               values[1] = TRUE;

            /* if variable in and-constraint1 is the negated variable of a normal linear variable, than we can add a
             * clique between the and-resultant and the normal linear variable, negated variables are not save in
             * cliquetables
             *
             * set r_1 = var1 * z; (z is some product)
             * var1 == ~var2
             *
             * if:
             * var1 + ~var1 <= 1;          r_1  
             *    0 +     1 <= 1             0   \
             *    1 +     0 <= 1   ==>  1 or 0    >   ==>    r_1 + var2 <= 1
             *    0 +     0 <= 1             0   /
             */
            if( values[0] != values[1] && var1 == var2 )
            {
               SCIP_CONS* newcons;
               SCIP_VAR* clqvars[2];
               char consname[SCIP_MAXSTRLEN];
               
               clqvars[0] = andress[c];
               clqvars[1] = values[1] ? var2 : SCIPvarGetNegatedVar(var2);
               assert(clqvars[1] != NULL);

               /* @todo: check whether it is better to only add the clique or to add the setppc constraint or do both */

               /* add clique */
               SCIP_CALL( SCIPaddClique(scip, clqvars, NULL, 2, cutoff, nchgbds) );
               if( *cutoff )
                  goto TERMINATE;

               (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s_clq_%s_%s", SCIPconsGetName(cons), SCIPvarGetName(clqvars[0]), SCIPvarGetName(clqvars[1]) );
               SCIP_CALL( SCIPcreateConsSetpack(scip, &newcons, consname, 2, clqvars, 
                     SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), 
                     FALSE, SCIPconsIsPropagated(cons),
                     SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons), 
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               SCIP_CALL( SCIPaddCons(scip, newcons) );
               SCIPdebugMessage("added a clique/setppc constraint <%s> \n", SCIPconsGetName(newcons));
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, newcons, NULL) ) );
               
               SCIP_CALL( SCIPreleaseCons(scip, &newcons) );
            }
            /* if a variable in an and-constraint is in a clique with another normal linear variable, we can add the
             * clique between the linear variable and the and-resultant 
             *
             * set r_1 = var1 * z; (z is some product)
             *
             * if:
             * var1 + var2 <= 1;          r_1
             *    0 +    1 <= 1             0   \
             *    1 +    0 <= 1   ==>  1 or 0    >   ==>    r_1 + var2 <= 1
             *    0 +    0 <= 1             0   /
             */
            if( SCIPvarsHaveCommonClique(var1, values[0], var2, values[1], TRUE) && (var1 != var2) )
            {
               SCIP_CONS* newcons;
               SCIP_VAR* clqvars[2];
               char consname[SCIP_MAXSTRLEN];

               clqvars[0] = andress[c];
               clqvars[1] = values[1] ? var2 : SCIPvarGetNegatedVar(var2);
               assert(clqvars[1] != NULL);

               /* @todo: check whether it is better to only add the clique or to add the setppc constraint or do both */

               /* add clique */
               SCIP_CALL( SCIPaddClique(scip, clqvars, NULL, 2, cutoff, nchgbds) );
               if( *cutoff )
                  goto TERMINATE;

               (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s_clq_%s_%s", SCIPconsGetName(cons), SCIPvarGetName(clqvars[0]), SCIPvarGetName(clqvars[1]) );
               SCIP_CALL( SCIPcreateConsSetpack(scip, &newcons, consname, 2, clqvars, 
                     SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), 
                     FALSE, SCIPconsIsPropagated(cons),
                     SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons), 
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               SCIP_CALL( SCIPaddCons(scip, newcons) );
               SCIPdebugMessage("added a clique/setppc constraint <%s> \n", SCIPconsGetName(newcons));
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, newcons, NULL) ) );
               
               SCIP_CALL( SCIPreleaseCons(scip, &newcons) );
            }
         }
      }
   }

   /* find cliques over variables which are in different and-constraints */
   for( c = nandress - 1; c > 0; --c )
   {
      CONSANDDATA* consanddata1;
      CONSANDDATA* consanddata2;
      SCIP_VAR** andvars1;
      int nandvars1;
      SCIP_VAR** andvars2;
      int nandvars2;
      
      consanddata1 = consdata->consanddatas[c];
      assert(consanddata1 != NULL);
      consanddata2 = consdata->consanddatas[c - 1];
      assert(consanddata2 != NULL);
      
      assert(SCIPgetResultantAnd(scip, consanddata1->cons) == andress[c]);
      assert(SCIPgetResultantAnd(scip, consanddata2->cons) == andress[c - 1]);

      /* choose correct variable array of consanddata object 1 */
      if( consanddata1->nnewvars > 0 )
      {
         andvars1 = consanddata1->newvars;
         nandvars1 = consanddata1->nnewvars;
      }
      else
      {
         andvars1 = consanddata1->vars;
         nandvars1 = consanddata1->nvars;
      }

      /* choose correct variable array of consanddata object 2 */
      if( consanddata2->nnewvars > 0 )
      {
         andvars2 = consanddata2->newvars;
         nandvars2 = consanddata2->nnewvars;
      }
      else
      {
         andvars2 = consanddata2->vars;
         nandvars2 = consanddata2->nvars;
      }

      /* compare both terms for finding new aggregated variables and new cliques */
      for( v1 = nandvars1 - 1; v1 >= 0; --v1 )
      {
         SCIP_VAR* var1;
         SCIP_Bool values[2];

         var1 = andvars1[v1];
         if( !SCIPvarIsActive(var1) && (!SCIPvarIsNegated(var1) || !SCIPvarIsActive(SCIPvarGetNegationVar(var1))) )
            continue;

         /* get active counterpart to check for common cliques */
         if( SCIPvarGetStatus(var1) == SCIP_VARSTATUS_NEGATED )
         {
            var1 = SCIPvarGetNegationVar(var1);
            values[0] = FALSE;
         }
         else
            values[0] = TRUE;

         for( v2 = nandvars2 - 1; v2 >= 0; --v2 )
         {
            SCIP_VAR* var2;

            var2 = andvars2[v2];
            if( !SCIPvarIsActive(var2) && (!SCIPvarIsNegated(var2) || !SCIPvarIsActive(SCIPvarGetNegationVar(var2))) )
               continue;
            
            /* get active counterpart to check for common cliques */
            if( SCIPvarGetStatus(var2) == SCIP_VARSTATUS_NEGATED )
            {
               var2 = SCIPvarGetNegationVar(var2);
               values[1] = FALSE;
            }
            else
               values[1] = TRUE;

            /* if a variable in and-constraint1 is the negated variable of a variable in and-constraint2, than we can
             * add a clique between both and-resultant, negated variables are not save in cliquetables
             *
             * set r_1 = var1 * z_1; (z_1 is some product)
             * set r_2 = var2 * z_2; (z_2 is some product)
             * var1 == ~var2
             *
             * if:
             * var1 + ~var1 <= 1;          r_1     r_2
             *    0 +     1 <= 1             0  1 or 0   \
             *    1 +     0 <= 1   ==>  1 or 0       0    >   ==>    r_1 + r_2 <= 1
             *    0 +     0 <= 1             0       0   /
             */
            if( values[0] != values[1] && var1 == var2 )
            {
               SCIP_CONS* newcons;
               SCIP_VAR* clqvars[2];
               char consname[SCIP_MAXSTRLEN];
               
               clqvars[0] = andress[c];
               clqvars[1] = andress[c - 1];

               /* @todo: check whether it is better to only add the clique or to add the setppc constraint or do both */

               /* add clique */
               SCIP_CALL( SCIPaddClique(scip, clqvars, NULL, 2, cutoff, nchgbds) );
               if( *cutoff )
                  goto TERMINATE;

               (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s_clq_%s_%s", SCIPconsGetName(cons), SCIPvarGetName(clqvars[0]), SCIPvarGetName(clqvars[1]) );
               SCIP_CALL( SCIPcreateConsSetpack(scip, &newcons, consname, 2, clqvars, 
                     SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), 
                     FALSE, SCIPconsIsPropagated(cons),
                     SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons), 
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               SCIP_CALL( SCIPaddCons(scip, newcons) );
               SCIPdebugMessage("added a clique/setppc constraint <%s> \n", SCIPconsGetName(newcons));
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, newcons, NULL) ) );
               
               SCIP_CALL( SCIPreleaseCons(scip, &newcons) );
            }
            /* if a variable in an and-constraint is in a clique with a variable in another and-constraint, we can add
             * the clique between both and-resultant
             *
             * let r_1 = var1 * z_1; (z_1 is some product)
             * let r_2 = var2 * z_2; (z_2 is some product)
             *
             * if:
             * var1 + var2 <= 1;          r_1     r_2
             *    0 +    1 <= 1             0  1 or 0   \
             *    1 +    0 <= 1   ==>  1 or 0       0    >   ==>    r_1 + r_2 <= 1
             *    0 +    0 <= 1             0       0   /
             */
            else if( SCIPvarsHaveCommonClique(var1, values[0], var2, values[1], TRUE) && (var1 != var2) )
            {
               SCIP_CONS* newcons;
               SCIP_VAR* clqvars[2];
               char consname[SCIP_MAXSTRLEN];

               clqvars[0] = andress[c];
               clqvars[1] = values[1] ? var2 : SCIPvarGetNegatedVar(var2);
               assert(clqvars[1] != NULL);

               /* @todo: check whether it is better to only add the clique or to add the setppc constraint or do both */

               /* add clique */
               SCIP_CALL( SCIPaddClique(scip, clqvars, NULL, 2, cutoff, nchgbds) );
               if( *cutoff )
                  goto TERMINATE;

               (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s_clq_%s_%s", SCIPconsGetName(cons), SCIPvarGetName(clqvars[0]), SCIPvarGetName(clqvars[1]) );
               SCIP_CALL( SCIPcreateConsSetpack(scip, &newcons, consname, 2, clqvars, 
                     SCIPconsIsInitial(cons), SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), 
                     FALSE, SCIPconsIsPropagated(cons),
                     SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons), 
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               SCIP_CALL( SCIPaddCons(scip, newcons) );
               SCIPdebugMessage("added a clique/setppc constraint <%s> \n", SCIPconsGetName(newcons));
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, newcons, NULL) ) );
               
               SCIP_CALL( SCIPreleaseCons(scip, &newcons) );
            }
         }
      }
   }

 TERMINATE:
   /* free temporary memory */
   SCIPfreeBufferArray(scip, &andcoefs);
   SCIPfreeBufferArray(scip, &andress);
   SCIPfreeBufferArray(scip, &lincoefs);
   SCIPfreeBufferArray(scip, &linvars);
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/** propagation method for pseudoboolean constraints */
static
SCIP_RETCODE propagateCons(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< knapsack constraint */
   SCIP_Bool*const       cutoff,             /**< pointer to store whether the node can be cut off */
   int*const             ndelconss           /**< pointer to count number of deleted constraints */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(cutoff != NULL);
   assert(ndelconss != NULL);

   *cutoff = FALSE;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->lincons != NULL);

   /* if linear constraint is redundant, than pseudoboolean constraint is redundant too */
   if( SCIPconsIsDeleted(consdata->lincons) )
   {
      SCIP_CALL( SCIPdelConsLocal(scip, cons) );
      ++(*ndelconss);
   }

   /* check if the constraint was already propagated */
   if( consdata->propagated )
      return SCIP_OKAY;

   /* mark the constraint propagated */
   consdata->propagated = TRUE;

   return SCIP_OKAY;
}

/* update and-constraint flags due to pseudoboolean constraint flags */
static
SCIP_RETCODE updateAndConss(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< pseudoboolean constraint */
   )
{
   CONSANDDATA** consanddatas;
   int nconsanddatas;
   SCIP_CONSDATA* consdata;
   int c;

   assert(scip != NULL);
   assert(cons != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   consanddatas = consdata->consanddatas;
   nconsanddatas = consdata->nconsanddatas;
   assert(nconsanddatas == 0 || consanddatas != NULL);

   if( !SCIPconsIsActive(cons) )
      return SCIP_OKAY;

   /* release and-constraints and change check flag of and-constraint */
   for( c = nconsanddatas - 1; c >= 0; --c )
   {
      SCIP_CONS* andcons;

      assert(consanddatas[c] != NULL);
      
      if( consanddatas[c]->deleted )
         continue;

      andcons = consanddatas[c]->cons;
      assert(andcons != NULL);

      SCIP_CALL( SCIPsetConsChecked(scip, andcons, SCIPconsIsChecked(cons)) );
   }
   
   return SCIP_OKAY;
}

/* delete unused information in constraint handler data */
static
SCIP_RETCODE correctConshdlrdata(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONSHDLRDATA*const conshdlrdata,     /**< pseudoboolean constraint handler data */
   int*const             ndelconss           /**< pointer to count number of deleted constraints */         
   )
{
   CONSANDDATA** allconsanddatas;
   CONSANDDATA* consanddata;
   int c;

   assert(scip != NULL);
   assert(conshdlrdata != NULL);
   assert(ndelconss != NULL);

   allconsanddatas = conshdlrdata->allconsanddatas;
   assert(allconsanddatas != NULL);
   assert(conshdlrdata->nallconsanddatas > 0);
   assert(conshdlrdata->nallconsanddatas <= conshdlrdata->sallconsanddatas);

   for( c = conshdlrdata->nallconsanddatas - 1; c >= 0; --c )
   {
      SCIP_VAR** tmpvars;
      int stmpvars;
      SCIP_CONS* cons;
      int v;
            
      consanddata = allconsanddatas[c];

      assert(consanddata->nvars == 0 || (consanddata->vars != NULL && consanddata->svars > 0));
      assert(consanddata->nnewvars == 0 || (consanddata->newvars != NULL && consanddata->snewvars > 0));

      if( consanddata->deleted )
      {
         assert(consanddata->vars == NULL);
         assert(consanddata->nvars == 0);
         assert(consanddata->svars == 0);
         assert(consanddata->newvars == NULL);
         assert(consanddata->nnewvars == 0);
         assert(consanddata->snewvars == 0);
         assert(consanddata->cons == NULL);

         continue;
      }

      /* if no variables are left, delete variables arrays */
      if( consanddata->nvars == 0 )
      {
         /* if we have no old variables, than also no new variables */
         assert(consanddata->nnewvars == 0);
         
         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->vars), consanddata->svars);
         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->newvars), consanddata->snewvars);

         /* delete and release and-constraint */
         SCIP_CALL( SCIPdelCons(scip, consanddata->cons) );
         SCIP_CALL( SCIPreleaseCons(scip, &(consanddata->cons)) );
         ++(*ndelconss);

         consanddata->nvars = 0;
         consanddata->svars = 0;
         consanddata->nnewvars = 0;
         consanddata->snewvars = 0;
         consanddata->deleted = TRUE;

         continue;
      }

      cons = consanddata->cons;
      assert(cons != NULL);

      /* if and-constraint is deleted, delete variables arrays */
      if( SCIPconsIsDeleted(cons) )
      {
         tmpvars = consanddata->vars;

         /* release all old variables */
         for( v = consanddata->nvars - 1; v >= 0; --v )
         {
            assert(tmpvars[v] != NULL);
            SCIP_CALL( SCIPreleaseVar(scip, &tmpvars[v]) );
         }

         tmpvars = consanddata->newvars;

         /* release all new variables */
         for( v = consanddata->nnewvars - 1; v >= 0; --v )
         {
            assert(tmpvars[v] != NULL);
            SCIP_CALL( SCIPreleaseVar(scip, &tmpvars[v]) );
         }

         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->vars), consanddata->svars);
         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->newvars), consanddata->snewvars)

         /* delete and release and-constraint */
         SCIP_CALL( SCIPdelCons(scip, consanddata->cons) );
         SCIP_CALL( SCIPreleaseCons(scip, &(consanddata->cons)) );
         ++(*ndelconss);

         consanddata->nvars = 0;
         consanddata->svars = 0;
         consanddata->nnewvars = 0;
         consanddata->snewvars = 0;
         consanddata->deleted = TRUE;

         continue;
      }
      
      /* if no new variables exist, we do not need to do anything here */
      if( consanddata->nnewvars == 0 )
         continue;

      tmpvars = consanddata->vars;
      /* release all variables */
      for( v = consanddata->nvars - 1; v >= 0; --v )
      {
         /* in original problem the variables was already deleted */
         assert(tmpvars[v] != NULL);
         SCIP_CALL( SCIPreleaseVar(scip, &tmpvars[v]) );
      }

      /* exchange newvars with old vars array */
      tmpvars = consanddata->vars;
      stmpvars = consanddata->svars;
      consanddata->vars = consanddata->newvars;
      consanddata->svars = consanddata->snewvars;
      consanddata->nvars = consanddata->nnewvars;
      consanddata->newvars = tmpvars;
      consanddata->snewvars = stmpvars;
      /* reset number of variables in newvars array */
      consanddata->nnewvars = 0;
   }      

   return SCIP_OKAY;
}

/* update the uses counter of consandata objects which are used in pseudoboolean constraint, which was deleted and
 * probably delete and-constraints 
 */
static
SCIP_RETCODE updateConsanddataUses(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   int*const             ndelconss           /**< pointer to store number of deleted constraints */
   )
{
   CONSANDDATA** consanddatas;
   int nconsanddatas;
   SCIP_CONSDATA* consdata;
   int c;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(ndelconss != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   consanddatas = consdata->consanddatas;
   nconsanddatas = consdata->nconsanddatas;
   assert(nconsanddatas > 0 && consanddatas != NULL);

#if 1
   if( nconsanddatas > 0 )
   {
      assert(consdata->andcoefs != NULL);

      for( c = nconsanddatas - 1; c >= 0; --c )
      {
         CONSANDDATA* consanddata;

         consanddata = consanddatas[c];
         assert(consanddata != NULL);
      
         if( consanddata->deleted )
            continue;

         SCIP_CALL( removeOldLocks(scip, cons, consanddata, consdata->andcoefs[c], consdata->lhs, consdata->rhs) );
      }
   }
#endif

#if 1
   for( c = nconsanddatas - 1; c >= 0; --c )
   {
      CONSANDDATA* consanddata;
            
      consanddata = consanddatas[c];
      assert(consanddata != NULL);
      assert(!(consanddatas[c]->deleted));

      assert(consanddata->nuses > 0);

      if( consanddata->nuses > 0 )
         --(consanddata->nuses);

      /* if data object is not used anymore, delete it */
      if( consanddata->nuses == 0 )
      {
         SCIP_VAR** tmpvars;
         int v;

         tmpvars = consanddata->vars;
            
         /* release all old variables */
         for( v = consanddata->nvars - 1; v >= 0; --v )
         {
            assert(tmpvars[v] != NULL);
            SCIP_CALL( SCIPreleaseVar(scip, &tmpvars[v]) );
         }

         tmpvars = consanddata->newvars;

         /* release all new variables */
         for( v = consanddata->nnewvars - 1; v >= 0; --v )
         {
            assert(tmpvars[v] != NULL);
            SCIP_CALL( SCIPreleaseVar(scip, &tmpvars[v]) );
         }

         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->vars), consanddata->svars);
         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->newvars), consanddata->snewvars);

         /* delete and release and-constraint */
         SCIP_CALL( SCIPdelCons(scip, consanddata->cons) );
         SCIP_CALL( SCIPreleaseCons(scip, &(consanddata->cons)) );
         ++(*ndelconss);

         consanddata->nvars = 0;
         consanddata->svars = 0;
         consanddata->nnewvars = 0;
         consanddata->snewvars = 0;
         consanddata->deleted = TRUE;
      }
   }
#endif

   return SCIP_OKAY;
}

/* try upgrading pseudoboolean logicor constraint to a linear constraint and/or remove possible and-constraints */
static
SCIP_RETCODE tryUpgradingLogicor(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_CONSHDLRDATA*const conshdlrdata,     /**< pseudoboolean constraint handler data */
   int*const             ndelconss,          /**< pointer to store number of deleted constraints */
   int*const             nfixedvars,         /**< pointer to store number of fixed variables */
   int*const             nchgcoefs,          /**< pointer to store number of changed coefficients constraints */
   int*const             nchgsides,          /**< pointer to store number of changed sides constraints */
   SCIP_Bool*const       cutoff              /**< pointer to store if a cutoff happened */
   )
{
   CONSANDDATA** consanddatas;
   int nconsanddatas;
   SCIP_CONSDATA* consdata;
   int c;
   int v;
   int v2;
   SCIP_VAR** eqvars;
   int neqvars;
   int nminvars;
   int nmaxvars;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(conshdlrdata != NULL);
   assert(ndelconss != NULL);
   assert(nfixedvars != NULL);
   assert(nchgcoefs != NULL);
   assert(nchgsides != NULL);
   assert(cutoff != NULL);
   assert(SCIPconsIsActive(cons));

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   consanddatas = consdata->consanddatas;
   nconsanddatas = consdata->nconsanddatas;
   assert(nconsanddatas > 0 && consanddatas != NULL);

   assert(consdata->lincons != NULL);
   assert(consdata->linconstype == SCIP_LOGICOR);

   assert(consanddatas[0] != NULL);
   assert(consanddatas[0]->cons != NULL);

   if( nconsanddatas == 1 )
   {
      /* if we have only one term left in the setppc constraint, the presolving should be done by the setppc constraint handler */
      if( consdata->nlinvars == 0 )
      {
         return SCIP_OKAY;
      }
      
      /* @todo: implement the following */

      /* for every old logicor constraint: 
       *     sum_i (x_i) + res >= 1 , with and-constraint of res as the resultant like res = y_1 * ... * y_n
       *  => sum_i (n * x_i) + sum_j=1^n y_j >= n 
       *
       * i.e. x_1 + x_2 + x_3 + x_4*x_5*x_6 >= 1
       *  => 3x_1 + 3x_2 + 3x_3 + x_4 + x_5 + x_6 >= 3
       */

      return SCIP_OKAY;
   }

#if 0
   if( consdata->nlinvars > 0 )
   {
      /* @todo: */
      return SCIP_OKAY;
   }
   assert(consdata->nlinvars == 0 && nconsanddatas > 1);
#endif

   c = nconsanddatas - 1;
   assert(!(consanddatas[c]->deleted));

   /* choose correct variable array */
   if( consanddatas[c]->nnewvars > 0 )
   {
      neqvars = consanddatas[c]->nnewvars;
      /* allocate temporary memory */
      SCIP_CALL( SCIPduplicateBufferArray(scip, &eqvars, consanddatas[c]->newvars, neqvars) );
   }
   else
   {
      neqvars = consanddatas[c]->nvars;
      /* allocate temporary memory */
      SCIP_CALL( SCIPduplicateBufferArray(scip, &eqvars, consanddatas[c]->vars, neqvars) );
   }
   nminvars = neqvars;
   nmaxvars = neqvars;
   assert(neqvars > 0 && eqvars != NULL);

#ifndef NDEBUG
   /* check that variables are sorted */
   for( v = neqvars - 1; v > 0; --v )
      assert(SCIPvarGetIndex(eqvars[v]) > SCIPvarGetIndex(eqvars[v - 1]));
#endif
   
   for( --c ; c >= 0; --c )
   {
      CONSANDDATA* consanddata;
      SCIP_VAR** vars;
      int nvars;
      int nneweqvars;

      consanddata = consanddatas[c];
      assert(consanddata != NULL);
      assert(!(consanddatas[c]->deleted));

      /* choose correct variable array to add locks for, we only add locks for now valid variables */
      if( consanddata->nnewvars > 0 )
      {
         vars = consanddata->newvars;
         nvars = consanddata->nnewvars;
      }
      else
      {
         vars = consanddata->vars;
         nvars = consanddata->nvars;
      }
      assert(nvars > 0 && vars != NULL);

#ifndef NDEBUG
      /* check that variables are sorted */
      for( v = nvars - 1; v > 0; --v )
         assert(SCIPvarGetIndex(vars[v]) > SCIPvarGetIndex(vars[v - 1]));
#endif

      /* update minimal number of variables in and-constraint */
      if( nvars < nminvars )
         nminvars = nvars;
      /* update maximal number of variables in and-constraint */
      else if( nvars > nmaxvars )
         nmaxvars = nvars;
      assert(nminvars > 0);
      assert(nminvars <= nmaxvars);

      /* now we only want to handle the easy case where nminvars == nmaxvars 
       * @todo: implement for the othercase too
       */
      if( nminvars < nmaxvars )
         break;

      nneweqvars = 0;
      for( v = 0, v2 = 0; v < neqvars && v2 < nvars; )
      {
         int index1;
         int index2;
         
         assert(eqvars[v] != NULL);
         assert(vars[v2] != NULL);
         index1 = SCIPvarGetIndex(eqvars[v]);
         index2 = SCIPvarGetIndex(vars[v2]);

         /* check which variables are still in all and-constraints */
         if( index1 < index2 )
            ++v;
         else if( index1 > index2 )
            ++v2;
         else
         {
            assert(index1 == index2);
            assert(nneweqvars <= v);

            if( nneweqvars < v )
               eqvars[nneweqvars] = eqvars[v];
            ++nneweqvars;
            ++v; 
            ++v2; 
         }
      }
      neqvars = nneweqvars;

      /* now we only want to handle the easy case where nminvars == neqvars + 1 
       * @todo: implement for the othercase too
       */
      if( nminvars > neqvars + 1 )
         break;

      if( neqvars == 0 )
         break;
   }

   /* if all and-constraints in pseudoboolean constraint have the same length and some equal variables we can upgrade
    * the linear constraint and fix all equal variables to 1
    */
   if( neqvars > 0 && nminvars == nmaxvars && nminvars == neqvars + 1 )
   {
      SCIP_CONS* lincons;
      SCIP_CONS* newcons;
      char newname[SCIP_MAXSTRLEN];
      SCIP_Real lhs;
      SCIP_Real rhs;
      SCIP_Bool infeasible;
      SCIP_Bool fixed;
      SCIP_Bool createcons;

      lhs = 1.0; 
      rhs = SCIPinfinity(scip); 
      createcons = TRUE;
      
#if 0
      /* if one and-constraint was completely contained in all other and-constraints, the new constraint will be
       * redundant 
       */
      if( neqvars == nminvars )
         createcons = FALSE; 
#endif

      lincons = consdata->lincons;

      if( createcons )
      {
         (void) SCIPsnprintf(newname, SCIP_MAXSTRLEN, "%s_upgraded", SCIPconsGetName(lincons));
         
         SCIP_CALL( SCIPcreateConsLinear(scip, &newcons, newname, 0, NULL, NULL, lhs, rhs,
               SCIPconsIsInitial(lincons), SCIPconsIsSeparated(lincons), SCIPconsIsEnforced(lincons), SCIPconsIsChecked(lincons),
               SCIPconsIsPropagated(lincons), SCIPconsIsLocal(lincons), SCIPconsIsModifiable(lincons),
               SCIPconsIsDynamic(lincons), SCIPconsIsRemovable(lincons), SCIPconsIsStickingAtNode(lincons)) );

         /* if createcons == TRUE add all variables which are not in the eqvars array to the new constraint with
          * coefficient 1.0 
          */
         for( c = nconsanddatas - 1; c >= 0; --c )
         {
            CONSANDDATA* consanddata;
            SCIP_VAR** vars;
            int nvars;
            
            consanddata = consanddatas[c];
            assert(consanddata != NULL);
            assert(!(consanddatas[c]->deleted));
            
            /* choose correct variable array to add locks for, we only add locks for now valid variables */
            if( consanddata->nnewvars > 0 )
            {
               vars = consanddata->newvars;
               nvars = consanddata->nnewvars;
            }
            else
            {
               vars = consanddata->vars;
               nvars = consanddata->nvars;
            }
            assert(nvars > 0 && vars != NULL);
            
            for( v = 0, v2 = 0; v < neqvars && v2 < nvars; )
            {
               int index1;
               int index2;
               
               assert(eqvars[v] != NULL);
               assert(vars[v2] != NULL);
               index1 = SCIPvarGetIndex(eqvars[v]);
               index2 = SCIPvarGetIndex(vars[v2]);
               
               /* all variables in eqvars array must exist in all and-constraints */
               assert(index1 >= index2);
               
               if( index1 > index2 )
               {
                  SCIP_CALL( SCIPaddCoefLinear(scip, newcons, vars[v2], 1.0) );
                  ++v2;
               }
               else
               {
                  assert(index1 == index2);
                  ++v;
                  ++v2;
               }
            }

            /* if we did not loop over all variables in the and-constraint, go on and fix variables */
            if( v2 < nvars )
            {
               assert(v == neqvars);
               for( ; v2 < nvars; ++v2)
               {
                  SCIP_CALL( SCIPaddCoefLinear(scip, newcons, vars[v2], 1.0) );
               }
            }
            assert(v == neqvars && v2 == nvars);
         }
      }

      /* if we have no normal linear variable in the linear constraint, we cann fix all equal variables, otherwise we have to add them with a coefficient of nconsanddatas */
      if( consdata->nlinvars == 0 )
      {
         /* fix all equal variable in logicor constraints which have to be one to fulfill the constraint */
         for( v = 0; v < neqvars; ++v )
         {
            /* fix the variable which cannot be one */
            SCIP_CALL( SCIPfixVar(scip, eqvars[v], 1.0, &infeasible, &fixed) );
            if( infeasible )
            {
               SCIPdebugMessage(" -> infeasible fixing\n");
               *cutoff = TRUE;
               goto TERMINATE;
            }
            if( fixed )
               ++(*nfixedvars);
         }
      }
      else
      {
         SCIP_VAR** vars;
         SCIP_Real* coefs;
         int nvars;
         SCIP_VAR** linvars;
         SCIP_Real* lincoefs;
         int nlinvars;

         /* add all equal variables */
         for( v = 0; v < neqvars; ++v )
         {
            SCIP_CALL( SCIPaddCoefLinear(scip, newcons, eqvars[v], (SCIP_Real)nconsanddatas) );
         }

         /* check number of linear variables */
         SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );
         assert(nvars == consdata->nlinvars + consdata->nconsanddatas);

         /* allocate temporary memory */
         SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
         SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
         SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
         SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );

         /* get variables and coefficients */ 
         SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
         assert(nvars == 0 || (vars != NULL && coefs != NULL));

#ifndef NDEBUG         
         /* all coefficients have to be 1 */
         for( v = 0; v < nvars; ++v )
            assert(SCIPisEQ(scip, coefs[v], 1.0));
#endif
         /* calculate all not artificial linear variables */
         SCIP_CALL( getLinVarsAndAndRess(scip, cons, vars, coefs, nvars, linvars, lincoefs, &nlinvars, NULL, NULL, NULL) );
         assert(nlinvars == consdata->nlinvars);
         
         /* add all old normal linear variables */
         for( v = 0; v < nlinvars; ++v )
         {
            SCIP_CALL( SCIPaddCoefLinear(scip, newcons, linvars[v], (SCIP_Real)(nconsanddatas * neqvars + 1)) );
         }

         /* reset left hand side to correct value */
         SCIP_CALL( SCIPchgLhsLinear(scip, newcons, (SCIP_Real)(nconsanddatas * neqvars + 1)) );

         /* free temporary memory */
         SCIPfreeBufferArray(scip, &lincoefs);
         SCIPfreeBufferArray(scip, &linvars);
         SCIPfreeBufferArray(scip, &coefs);
         SCIPfreeBufferArray(scip, &vars);
      }

      /* add and release new constraint */
      if( createcons )
      {
         SCIP_CALL( SCIPaddCons(scip, newcons) );

         SCIPdebugMessage("created upgraded linear constraint:\n");
         SCIPdebugMessage("old -> ");
         SCIPdebug( SCIP_CALL( SCIPprintCons(scip, lincons, NULL) ) );
         SCIPdebugMessage("new -> ");
         SCIPdebug( SCIP_CALL( SCIPprintCons(scip, newcons, NULL) ) );

         SCIP_CALL( SCIPreleaseCons(scip, &newcons) );
      }

      /* delete old constraints */
      SCIP_CALL( SCIPdelCons(scip, lincons) );
      SCIP_CALL( SCIPdelCons(scip, cons) );
      ++(*ndelconss);
   }

 TERMINATE:
   /* free temporary memory */
   SCIPfreeBufferArray(scip, &eqvars);

   return SCIP_OKAY;
}

/* try upgrading pseudoboolean setppc constraint to a linear constraint and/or remove possible and-constraints */
static
SCIP_RETCODE tryUpgradingSetppc(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_CONSHDLRDATA*const conshdlrdata,     /**< pseudoboolean constraint handler data */
   int*const             ndelconss,          /**< pointer to store number of deleted constraints */
   int*const             nfixedvars,         /**< pointer to store number of fixed variables */
   int*const             nchgcoefs,          /**< pointer to store number of changed coefficients constraints */
   int*const             nchgsides,          /**< pointer to store number of changed sides constraints */
   SCIP_Bool*const       cutoff              /**< pointer to store if a cutoff happened */
   )
{
   CONSANDDATA** consanddatas;
   int nconsanddatas;
   SCIP_CONSDATA* consdata;
   SCIP_SETPPCTYPE type;
   int c;
   int v;
   int v2;
   SCIP_VAR** eqvars;
   int neqvars;
   int nminvars;
   int nmaxvars;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(conshdlrdata != NULL);
   assert(ndelconss != NULL);
   assert(nfixedvars != NULL);
   assert(nchgcoefs != NULL);
   assert(nchgsides != NULL);
   assert(cutoff != NULL);
   assert(SCIPconsIsActive(cons));

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   consanddatas = consdata->consanddatas;
   nconsanddatas = consdata->nconsanddatas;
   assert(nconsanddatas > 0 && consanddatas != NULL);

   assert(consdata->lincons != NULL);
   assert(consdata->linconstype == SCIP_SETPPC);

   type = SCIPgetTypeSetppc(scip, consdata->lincons);
   
   switch( type )
   {
   case SCIP_SETPPCTYPE_PARTITIONING:
   case SCIP_SETPPCTYPE_PACKING:
      break;
   case SCIP_SETPPCTYPE_COVERING:
      return SCIP_OKAY;
   default:
      SCIPerrorMessage("unknown setppc type\n");
      return SCIP_INVALIDDATA;
   }

   assert(consanddatas[0] != NULL);
   assert(consanddatas[0]->cons != NULL);

   if( nconsanddatas == 1 )
   {
      /* if we have only one term left in the setppc constraint, the presolving should be done by the setppc constraint handler */
      if( consdata->nlinvars == 0 )
      {
         return SCIP_OKAY;
      }
      
      /* @todo: implement the following */

      /* for every old set packing constraint: 
       *     sum_i (x_i) + res <= 1 , with and-constraint of res as the resultant like res = y_1 * ... * y_n
       *  => sum_i (n * x_i) + sum_j=1^n y_j <= n + n-1
       *
       * i.e. x_1 + x_2 + x_3 + x_4*x_5*x_6 <= 1
       *  => 3x_1 + 3x_2 + 3x_3 + x_4 + x_5 + x_6 <= 5
       */

      /* for every old set partitioning constraint: 
       *     sum_i (x_i) + res = 1 , with the corresponding and-constraint of res like 
       *                             res = y_1 * ... * y_n and all y_j != x_i forall j and i
       *
       *  => sum_i (n * x_i) + sum_j=1^n y_j = n
       *
       * i.e. x_1 + x_2 + x_3 + x_4*x_5*x_6 = 1
       *  => 3x_1 + 3x_2 + 3x_3 + x_4 + x_5 + x_6 = 3
       *
       * but if i.e. x_4 = x_1 in the original constraint the vector (x_1,x_2,x_3,x_4,x_5,x_6) = (1,0,0,1,0,0) is a
       * solution but not in the new constraint anymore
       */

      return SCIP_OKAY;
   }

   if( consdata->nlinvars > 0 )
   {
      /* @todo: */
      return SCIP_OKAY;
   }
   assert(consdata->nlinvars == 0 && nconsanddatas > 1);

   c = nconsanddatas - 1;
   assert(!(consanddatas[c]->deleted));

   /* choose correct variable array */
   if( consanddatas[c]->nnewvars > 0 )
   {
      neqvars = consanddatas[c]->nnewvars;
      /* allocate temporary memory */
      SCIP_CALL( SCIPduplicateBufferArray(scip, &eqvars, consanddatas[c]->newvars, neqvars) );
   }
   else
   {
      neqvars = consanddatas[c]->nvars;
      /* allocate temporary memory */
      SCIP_CALL( SCIPduplicateBufferArray(scip, &eqvars, consanddatas[c]->vars, neqvars) );
   }
   nminvars = neqvars;
   nmaxvars = neqvars;
   assert(neqvars > 0 && eqvars != NULL);

#ifndef NDEBUG
   /* check that variables are sorted */
   for( v = neqvars - 1; v > 0; --v )
      assert(SCIPvarGetIndex(eqvars[v]) > SCIPvarGetIndex(eqvars[v - 1]));
#endif
   
   for( --c ; c >= 0; --c )
   {
      CONSANDDATA* consanddata;
      SCIP_VAR** vars;
      int nvars;
      int nneweqvars;

      consanddata = consanddatas[c];
      assert(consanddata != NULL);
      assert(!(consanddatas[c]->deleted));

      /* choose correct variable array to add locks for, we only add locks for now valid variables */
      if( consanddata->nnewvars > 0 )
      {
         vars = consanddata->newvars;
         nvars = consanddata->nnewvars;
      }
      else
      {
         vars = consanddata->vars;
         nvars = consanddata->nvars;
      }
      assert(nvars > 0 && vars != NULL);

#ifndef NDEBUG
      /* check that variables are sorted */
      for( v = nvars - 1; v > 0; --v )
         assert(SCIPvarGetIndex(vars[v]) > SCIPvarGetIndex(vars[v - 1]));
#endif

      /* update minimal number of variables in and-constraint */
      if( nvars < nminvars )
         nminvars = nvars;
      /* update maximal number of variables in and-constraint */
      else if( nvars > nmaxvars )
         nmaxvars = nvars;
      assert(nminvars > 0);
      assert(nminvars <= nmaxvars);

      /* now we only want to handle the easy case where nminvars == nmaxvars 
       * @todo: implement for the othercase too
       */
      if( nminvars < nmaxvars )
         break;

      nneweqvars = 0;
      for( v = 0, v2 = 0; v < neqvars && v2 < nvars; )
      {
         int index1;
         int index2;
         
         assert(eqvars[v] != NULL);
         assert(vars[v2] != NULL);
         index1 = SCIPvarGetIndex(eqvars[v]);
         index2 = SCIPvarGetIndex(vars[v2]);

         /* check which variables are still in all and-constraints */
         if( index1 < index2 )
            ++v;
         else if( index1 > index2 )
            ++v2;
         else
         {
            assert(index1 == index2);
            assert(nneweqvars <= v);

            if( nneweqvars < v )
               eqvars[nneweqvars] = eqvars[v];
            ++nneweqvars;
            ++v; 
            ++v2; 
         }
      }
      neqvars = nneweqvars;

      /* now we only want to handle the easy case where nminvars == neqvars + 1 
       * @todo: implement for the othercase too
       */
      if( nminvars > neqvars + 1 )
         break;

      if( neqvars == 0 )
         break;
   }

   /* if all and-constraints in pseudoboolean constraint have the same length and some equal variables we can upgrade
    * the linear constraint and fix some variables in setpartitioning case
    */
   if( neqvars > 0 && nminvars == nmaxvars && nminvars == neqvars + 1 )
   {
      SCIP_CONS* lincons;
      SCIP_CONS* newcons;
      char newname[SCIP_MAXSTRLEN];
      SCIP_Real lhs;
      SCIP_Real rhs;
      SCIP_Bool infeasible;
      SCIP_Bool fixed;
      SCIP_Bool createcons;

      /* determine new sides of linear constraint */
      if( type == SCIP_SETPPCTYPE_PARTITIONING )
      {
         lhs = 1.0; 
         rhs = 1.0; 
      }
      else
      {
         assert(type == SCIP_SETPPCTYPE_PACKING);
         lhs = -SCIPinfinity(scip); 
         rhs = 1.0; 
      }

#if 0
      /* if one and-constraint was completely contained in all other and-constraints, we have to reduced the right hand
       * side by 1
       */
      if( neqvars == nminvars )
         rhs -= 1.0; 
#endif

      createcons = SCIPisLE(scip, lhs, rhs);
      assert(createcons || type == SCIP_SETPPCTYPE_PARTITIONING);

      lincons = consdata->lincons;

      if( createcons )
      {
         (void) SCIPsnprintf(newname, SCIP_MAXSTRLEN, "%s_upgraded", SCIPconsGetName(lincons));
         
         SCIP_CALL( SCIPcreateConsLinear(scip, &newcons, newname, 0, NULL, NULL, lhs, rhs,
               SCIPconsIsInitial(lincons), SCIPconsIsSeparated(lincons), SCIPconsIsEnforced(lincons), SCIPconsIsChecked(lincons),
               SCIPconsIsPropagated(lincons), SCIPconsIsLocal(lincons), SCIPconsIsModifiable(lincons),
               SCIPconsIsDynamic(lincons), SCIPconsIsRemovable(lincons), SCIPconsIsStickingAtNode(lincons)) );
      }
      
      /* if createcons == TRUE add all variables which are not in the eqvars array to the new constraint with
       * coefficient 1.0 
       *
       * otherwise (if createcons == FALSE) fix all variables to zero which are not in the eqvars array and if we have a
       * set partitioning constraint 
       */
      for( c = nconsanddatas - 1; c >= 0; --c )
      {
         CONSANDDATA* consanddata;
         SCIP_VAR** vars;
         int nvars;
            
         consanddata = consanddatas[c];
         assert(consanddata != NULL);
         assert(!(consanddatas[c]->deleted));
            
         /* choose correct variable array to add locks for, we only add locks for now valid variables */
         if( consanddata->nnewvars > 0 )
         {
            vars = consanddata->newvars;
            nvars = consanddata->nnewvars;
         }
         else
         {
            vars = consanddata->vars;
            nvars = consanddata->nvars;
         }
         assert(nvars > 0 && vars != NULL);

         for( v = 0, v2 = 0; v < neqvars && v2 < nvars; )
         {
            int index1;
            int index2;
               
            assert(eqvars[v] != NULL);
            assert(vars[v2] != NULL);
            index1 = SCIPvarGetIndex(eqvars[v]);
            index2 = SCIPvarGetIndex(vars[v2]);

            /* all variables in eqvars array must exist in all and-constraints */
            assert(index1 >= index2);
               
            if( index1 > index2 )
            {
               if( createcons )
               {
                  SCIP_CALL( SCIPaddCoefLinear(scip, newcons, vars[v2], 1.0) );
               }
               else
               {
                  assert(type == SCIP_SETPPCTYPE_PARTITIONING);

                  /* fix the variable which cannot be one */
                  SCIP_CALL( SCIPfixVar(scip, vars[v2], 0.0, &infeasible, &fixed) );
                  if( infeasible )
                  {
                     SCIPdebugMessage(" -> infeasible fixing\n");
                     *cutoff = TRUE;
                     goto TERMINATE;
                  }
                  if( fixed )
                     ++(*nfixedvars);
               }
               ++v2;
            }
            else
            {
               assert(index1 == index2);

               ++v;
               ++v2;
            }
         }

         /* if we did not loop over all variables in the and-constraint, go on and fix variables */
         if( v2 < nvars )
         {
            assert(v == neqvars);
            for( ; v2 < nvars; ++v2)
            {
               if( createcons )
               {
                  SCIP_CALL( SCIPaddCoefLinear(scip, newcons, vars[v2], 1.0) );
               }
               else
               {
                  assert(type == SCIP_SETPPCTYPE_PARTITIONING);

                  /* fix the variable which cannot be one */
                  SCIP_CALL( SCIPfixVar(scip, vars[v2], 0.0, &infeasible, &fixed) );
                  if( infeasible )
                  {
                     SCIPdebugMessage(" -> infeasible fixing\n");
                     *cutoff = TRUE;
                     goto TERMINATE;
                  }
                  if( fixed )
                     ++(*nfixedvars);
               }
            }
         }
         assert(v == neqvars && v2 == nvars);
      }

      /* fix all equal variable in set-partitioning constraints which have to be one, in set-packing constraint we have
       * to add these variable with a coeffcient as big as (nconsanddatas - 1) 
       */
      for( v = 0; v < neqvars; ++v )
      {
         if( type == SCIP_SETPPCTYPE_PARTITIONING )
         {
            /* fix the variable which cannot be one */
            SCIP_CALL( SCIPfixVar(scip, eqvars[v], 1.0, &infeasible, &fixed) );
            if( infeasible )
            {
               SCIPdebugMessage(" -> infeasible fixing\n");
               *cutoff = TRUE;
               goto TERMINATE;
            }
            if( fixed )
               ++(*nfixedvars);
         }
         else 
         {
            assert(type == SCIP_SETPPCTYPE_PACKING);
            SCIP_CALL( SCIPaddCoefLinear(scip, newcons, eqvars[v], (SCIP_Real)(nconsanddatas - 1)) );
         }
      }

      /* correct right hand side for set packing constraint */
      if( type == SCIP_SETPPCTYPE_PACKING )
      {
         assert(SCIPisEQ(scip, rhs, 1.0));
         assert(createcons);
         
         SCIP_CALL( SCIPchgRhsLinear(scip, newcons, rhs + (SCIP_Real)((nconsanddatas - 1) * neqvars)) );
      }

      /* add and release new constraint */
      if( createcons )
      {
         SCIP_CALL( SCIPaddCons(scip, newcons) );

         SCIPdebugMessage("created upgraded linear constraint:\n");
         SCIPdebugMessage("old -> ");
         SCIPdebug( SCIP_CALL( SCIPprintCons(scip, lincons, NULL) ) );
         SCIPdebugMessage("new -> ");
         SCIPdebug( SCIP_CALL( SCIPprintCons(scip, newcons, NULL) ) );

         SCIP_CALL( SCIPreleaseCons(scip, &newcons) );
      }

      /* delete old constraints */
      SCIP_CALL( SCIPdelCons(scip, lincons) );
      SCIP_CALL( SCIPdelCons(scip, cons) );
      ++(*ndelconss);
   }

 TERMINATE:
   /* free temporary memory */
   SCIPfreeBufferArray(scip, &eqvars);

   return SCIP_OKAY;
}

/* try upgrading pseudoboolean constraint to a linear constraint and/or remove possible and-constraints */
static
SCIP_RETCODE tryUpgrading(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_CONSHDLRDATA*const conshdlrdata,     /**< pseudoboolean constraint handler data */
   int*const             ndelconss,         /**< pointer to store number of upgraded constraints */
   int*const             nfixedvars,         /**< pointer to store number of fixed variables */
   int*const             nchgcoefs,          /**< pointer to store number of changed coefficients constraints */
   int*const             nchgsides,          /**< pointer to store number of changed sides constraints */
   SCIP_Bool*const       cutoff              /**< pointer to store if a cutoff happened */
   )
{
   CONSANDDATA** consanddatas;
   SCIP_CONSDATA* consdata;
   int nvars;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(conshdlrdata != NULL);
   assert(ndelconss != NULL);
   assert(nfixedvars != NULL);
   assert(nchgcoefs != NULL);
   assert(nchgsides != NULL);
   assert(cutoff != NULL);
   assert(SCIPconsIsActive(cons));

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->lincons != NULL);

   consanddatas = consdata->consanddatas;
   assert(consdata->nconsanddatas == 0 || consanddatas != NULL);

   /* if no consanddata-objects in pseudoboolean constraint are left, create the corresponding linear constraint */
   if( consdata->nconsanddatas == 0 )
   {
      if( consdata->linconstype == SCIP_LINEAR )
      {
         SCIP_CALL( SCIPsetUpgradeConsLinear(scip, consdata->lincons, TRUE) );
      }
      /* @TODO: maybe it is better to create everytime a standard linear constraint instead of letting the special
       *        linear constraint stay 
       */
      SCIP_CALL( SCIPdelCons(scip, cons) );
      ++(*ndelconss);

      return SCIP_OKAY;
   }

   /* check number of linear variables */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );
   assert(consdata->nlinvars + consdata->nconsanddatas == nvars);

   switch( consdata->linconstype )
   {
   case SCIP_LINEAR:
      break;
   case SCIP_LOGICOR:
      SCIP_CALL( tryUpgradingLogicor(scip, cons, conshdlrdata, ndelconss, nfixedvars, nchgcoefs, nchgsides, cutoff) );
      break;
   case SCIP_KNAPSACK:
      break;
   case SCIP_SETPPC:
      SCIP_CALL( tryUpgradingSetppc(scip, cons, conshdlrdata, ndelconss, nfixedvars, nchgcoefs, nchgsides, cutoff) );
      break;
#if 0
   case SCIP_EQKNAPSACK:
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   if( SCIPconsIsDeleted(cons) )
   {
      /* update the uses counter of consandata objects which are used in pseudoboolean constraint, which was deleted and
       * probably delete and-constraints 
       */
      SCIP_CALL( updateConsanddataUses(scip, cons, ndelconss) );
   }

   consdata->upgradetried = TRUE;
  
   return SCIP_OKAY;
}

#ifdef SCIP_DEBUG 
/** check constraint consistency */
static
SCIP_RETCODE checkConsConsistency(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< pseudoboolean constraint */
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;
   SCIP_VAR** linvars;
   SCIP_Real* lincoefs;
   int nlinvars;
   SCIP_VAR** andress;
   SCIP_Real* andcoefs;
   int nandress;
   SCIP_Bool* alreadyfound;
   SCIP_VAR* res;
   int c;
   int v;
   SCIP_Real newlhs;
   SCIP_Real newrhs;
   
   assert(scip != NULL);
   assert(cons != NULL);
   assert(SCIPconsIsActive(cons));

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* check standard pointers and sizes */
   assert(consdata->lincons != NULL);
   assert(SCIPconsIsActive(consdata->lincons));
   assert(consdata->linconstype > SCIP_INVALIDCONS);
   assert(consdata->andconss != NULL);
   assert(consdata->nandconss > 0);
   assert(consdata->nandconss <= consdata->sandconss);

   /* get sides of linear constraint */
   SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &newlhs, &newrhs) );
   assert(!SCIPisInfinity(scip, newlhs));
   assert(!SCIPisInfinity(scip, -newrhs));
   assert(SCIPisLE(scip, newlhs, newrhs));
   assert(SCIPisEQ(scip, newrhs, consdata->rhs);
   assert(SCIPisEQ(scip, newlhs, consdata->lhs);

   /* check number of linear variables */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );
   assert(nvars == consdata->nlinvars + consdata->nconsanddatas);

   /* get temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andress, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &alreadyfound, nvars) );
   BMSclearMemoryArray(alreadyfound, nvars);

   /* get variables and coefficients */ 
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
   assert(nvars == 0 || (vars != NULL && coefs != NULL));

   /* calculate all not artificial linear variables and all artificial and-resultants which will be ordered like the
    * 'consanddatas' such that the and-resultant of the and-constraint is the and-resultant in the 'andress' array
    * afterwards 
    */
   SCIP_CALL( getLinVarsAndAndRess(scip, cons, vars, coefs, nvars, linvars, lincoefs, &nlinvars, andress, andcoefs, &nandress) );
   assert(nlinvars == consdata->nlinvars);

   for( v = nandress - 1; v >= 0; --v )
   {
      for(c = consdata->nandconss - 1; c >= 0; --c )
      {
         assert(consdata->andconss[c] != NULL);
         res = SCIPgetResultantAnd(scip, consdata->andconss[c]);
         assert(res != NULL);
         if( res == andress[v] )
         {
            /* resultant should be either active or a negated variable of an active one */
            assert(SCIPvarIsActive(res) || (SCIPvarIsNegated(res) && SCIPvarIsActive(SCIPvarGetNegationVar(res))));
            assert(!alreadyfound[c]);
            
            /* all and-resultants should be merged, so it is only allowed that each variable exists one time */
            alreadyfound[c] = TRUE;
            break;
         }
      }
   }

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &alreadyfound);
   SCIPfreeBufferArray(scip, &andcoefs);
   SCIPfreeBufferArray(scip, &andress);
   SCIPfreeBufferArray(scip, &lincoefs);
   SCIPfreeBufferArray(scip, &linvars);
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}
#endif


/*
 * Callback methods of constraint handler
 */

/** copy method for constraint handler plugins (called when SCIP copies plugins) */
static
SCIP_DECL_CONSHDLRCOPY(conshdlrCopyPseudoboolean)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   
   /* call inclusion method of constraint handler */
   SCIP_CALL( SCIPincludeConshdlrPseudoboolean(scip) );

   *valid = TRUE;

   return SCIP_OKAY;
}

/** destructor of constraint handler to free constraint handler data (called when SCIP is exiting) */
static
SCIP_DECL_CONSFREE(consFreePseudoboolean)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   /* free constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIP_CALL( conshdlrdataFree(scip, &conshdlrdata) );
   
   SCIPconshdlrSetData(conshdlr, NULL);

   return SCIP_OKAY;
}


/** initialization method of constraint handler (called after problem was transformed) */
static
SCIP_DECL_CONSINIT(consInitPseudoboolean)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   int c;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* check each constraint and get transformed constraints */
   for( c = conshdlrdata->nallconsanddatas - 1; c >= 0; --c )
   {
      SCIP_CONS* andcons;
      SCIP_VAR** vars;
      int nvars;
      int v;
      SCIP_VAR* resultant;
      
      assert(conshdlrdata->allconsanddatas[c] != NULL);
      assert(conshdlrdata->allconsanddatas[c]->newvars == NULL);

      vars = conshdlrdata->allconsanddatas[c]->vars;
      nvars = conshdlrdata->allconsanddatas[c]->nvars;
      assert(vars != NULL || nvars == 0);
         
      /* get transformed variables */
      SCIP_CALL( SCIPgetTransformedVars(scip, nvars, vars, vars) );

      /* resort variables in transformed problem, because the order might change while tranforming */
      SCIPsortPtr((void**)vars, SCIPvarComp, nvars);
      
      /* do not capture after restart */
      if( SCIPgetNRuns(scip) < 1 )
      {
         /* capture all variables */
         for( v = nvars - 1; v >= 0; --v )
         {
            SCIP_CALL( SCIPcaptureVar(scip, vars[v]) );
         }
      }

      andcons = conshdlrdata->allconsanddatas[c]->cons;
      assert(andcons != NULL);

      conshdlrdata->allconsanddatas[c]->origcons = andcons;

      /* in a restart the constraints might already be transformed */
      if( !SCIPconsIsTransformed(andcons) )
      {
         SCIP_CONS* transcons;
         
         SCIP_CALL( SCIPgetTransformedCons(scip, andcons, &transcons) );

         if( transcons == NULL )
         {
            conshdlrdata->allconsanddatas[c]->origcons = NULL;
            continue;
         }

         assert( transcons != NULL );
         conshdlrdata->allconsanddatas[c]->cons = transcons;
         
         resultant = SCIPgetResultantAnd(scip, transcons);
         /* insert new mapping */
         assert(!SCIPhashmapExists(conshdlrdata->hashmap, (void*)resultant));
         SCIP_CALL( SCIPhashmapInsert(conshdlrdata->hashmap, (void*)resultant, (void*)(conshdlrdata->allconsanddatas[c])) );

         /* capture constraint */
         SCIP_CALL( SCIPcaptureCons(scip, conshdlrdata->allconsanddatas[c]->cons) );
      }
      resultant = SCIPgetResultantAnd(scip, conshdlrdata->allconsanddatas[c]->cons);
      assert(SCIPhashmapExists(conshdlrdata->hashmap, (void*)resultant));
   }

   /* check each constraint and get transformed constraints */
   for( c = nconss - 1; c >= 0; --c )
   {
      SCIP_CONSDATA* consdata;
      SCIP_CONS* cons;
#if 0
      int a;
      SCIP_Real rhs;
      SCIP_Real lhs;
#endif
      assert(conss != NULL);

      cons = conss[c];
      assert(cons != NULL);
      assert(SCIPconsIsTransformed(cons));

      consdata = SCIPconsGetData(cons);
      assert(consdata != NULL);

      /* if not happend already, get transformed linear constraint */
      assert(consdata->lincons != NULL);
      assert(consdata->linconstype > SCIP_INVALIDCONS);

      /* in a restart the linear constraint might already be transformed */
      if( !SCIPconsIsTransformed(consdata->lincons) )
      {
         SCIP_CONS* transcons;

         SCIP_CALL( SCIPgetTransformedCons(scip, consdata->lincons, &transcons) );
         assert( transcons != NULL );

         /* we want to check all tranformed constraints */
         SCIP_CALL( SCIPsetConsChecked(scip, transcons, SCIPconsIsChecked(cons)) );

         SCIP_CALL( SCIPcaptureCons(scip, transcons) );
         consdata->lincons = transcons;
      }

#if 0
      lhs = consdata->lhs;
      rhs = consdata->rhs;

      assert(consdata->nconsanddatas == 0 || (consdata->consanddatas != NULL && consdata->andcoefs != NULL));
      for( a = consdata->nconsanddatas - 1; a >= 0; --a )
      {
         /* add rounding locks due to old variables in consanddata object */
         SCIP_CALL( lockRoundingAndCons(scip, cons, consdata->consanddatas[a], consdata->andcoefs[a], lhs, rhs) );
      }
#endif
   }

   return SCIP_OKAY;
}

/** deinitialization method of constraint handler (called before transformed problem is freed) */
static
SCIP_DECL_CONSEXIT(consExitPseudoboolean)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   CONSANDDATA** allconsanddatas;
   int c;
   int nallconsanddatas;
   int sallconsanddatas;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(nconss == 0 || conss != NULL);

#if 0 //not possible because we can't call SCIPaddVarLocks in SCIP_STAGE_TRANSFORMED
   /* trying to correct rounding locks of every single variable corresponding to the and-constraints */
   for( c = nconss - 1; c >= 0; --c )
   {
      SCIP_CONSDATA* consdata;
      SCIP_Real lhs;
      SCIP_Real rhs;
      SCIP_Bool haslhs;
      SCIP_Bool hasrhs;
      int d;

      assert(conss[c] != NULL);
            
      consdata = SCIPconsGetData(conss[c]);
      assert(consdata != NULL);

      lhs = consdata->lhs;
      rhs = consdata->rhs;
      assert(!SCIPisInfinity(scip, lhs));
      assert(!SCIPisInfinity(scip, -rhs));
      assert(SCIPisLE(scip, lhs, rhs));
      
      haslhs = !SCIPisInfinity(scip, -lhs);
      hasrhs = !SCIPisInfinity(scip, rhs);
   
      for( d = consdata->nconsanddatas - 1; d >= 0; --d )
      {
         SCIP_VAR* andres;
         SCIP_VAR** andvars;
         SCIP_Real val;
         int nandvars;
         SCIP_CONS* andcons;
         CONSANDDATA* consanddata;
         int v;

         consanddata = consdata->consanddatas[c];
         
         if( consanddata->deleted )
            continue;

         assert(consanddata != NULL);
         andcons = consanddata->cons;
         assert(andcons != NULL);
         assert(consanddata->nnewvars == 0);

         andvars = consanddata->vars;
         nandvars = consanddata->nvars;

         /* probably we need to store the resultant too, now it's not possible to remove the resultant from the and-constraint */
         andres = SCIPgetResultantAnd(scip, andcons);
         assert(nandvars == 0 || andvars != NULL);
         assert(andres != NULL);
         val = consdata->andcoefs[c];
         
         /* lock variables */
         if( SCIPisPositive(scip, val) )
         {
            if( haslhs )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), -1, 0);
                  SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], -1, 0) );
               }
               //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), -1, -1);
               SCIP_CALL( SCIPaddVarLocks(scip, andres, -1, -1) );
            }
            if( hasrhs )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), 0, -1);
                  SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], 0, -1) );
               }
               /* don't double the locks on the and-resultant */
               if( !haslhs )
               {
                  //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), -1, -1);
                  SCIP_CALL( SCIPaddVarLocks(scip, andres, -1, -1) );
               }
            }
         }
         else
         {
            if( haslhs )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), 0, -1);
                  SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], 0, -1) );
            }
               //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), -1, -1);
               SCIP_CALL( SCIPaddVarLocks(scip, andres, -1, -1) );
            }
            if( hasrhs )
            {
               for( v = nandvars - 1; v >= 0; --v )
               {
                  //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), -1, 0);
                  SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], -1, 0) );
               }
               /* don't double the locks on the and-resultant */
               if( !haslhs )
               {
                  //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), -1, -1);
                  SCIP_CALL( SCIPaddVarLocks(scip, andres, -1, -1) );
               }
            }
         }
      }
   }
#endif

   /* free constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   allconsanddatas = conshdlrdata->allconsanddatas;
   nallconsanddatas = conshdlrdata->nallconsanddatas;
   sallconsanddatas = conshdlrdata->sallconsanddatas;
 
   /* release and-constraints */
   for( c = nallconsanddatas - 1; c >= 0; --c )
   {
      if( allconsanddatas[c] != NULL && !(allconsanddatas[c]->deleted) )
      {
         SCIP_VAR** vars;
         int nvars;
         int v;

         vars = allconsanddatas[c]->vars;
         nvars = allconsanddatas[c]->nvars;
         assert(vars != NULL || nvars == 0);
         
         /* release all variables */
         for( v = nvars - 1; v >= 0; --v )
         {
            /* in original problem the variables was already deleted */
            assert(vars[v] != NULL);
            SCIP_CALL( SCIPreleaseVar(scip, &vars[v]) );
         }
         
         assert(allconsanddatas[c]->nnewvars == 0);

         /* in original problem the constraint was already deleted */
         assert(allconsanddatas[c]->cons != NULL);
         SCIP_CALL( SCIPreleaseCons(scip, &(allconsanddatas[c]->cons)) );
      }
      if( allconsanddatas[c]->origcons != NULL )
      { 
         SCIP_CALL( SCIPreleaseCons(scip, &(allconsanddatas[c]->origcons)) );
      }
   }

#if 0
   /* clear old conshdlrdata */
   SCIP_CALL( conshdlrdataFree(scip, &conshdlrdata) );

   /* recreate conshdlrdata */
   SCIP_CALL( conshdlrdataCreate(scip, &conshdlrdata) );
   SCIPconshdlrSetData(conshdlr, conshdlrdata);
#else 
   /* clear constraint handler data */
   SCIP_CALL( conshdlrdataClear(scip, &conshdlrdata) );
#endif

   return SCIP_OKAY;
}

/** presolving initialization method of constraint handler (called when presolving is about to begin) */
static
SCIP_DECL_CONSINITPRE(consInitprePseudoboolean)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   int c;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

#if 0
   /* check each constraint and get transformed constraints */
   for( c = conshdlrdata->nallconsanddatas - 1; c >= 0; --c )
   {
      SCIP_CONS* andcons;
      SCIP_VAR** vars;
      int nvars;
      int v;
      SCIP_VAR* resultant;
      
      assert(conshdlrdata->allconsanddatas[c] != NULL);
      assert(conshdlrdata->allconsanddatas[c]->newvars == NULL);

      vars = conshdlrdata->allconsanddatas[c]->vars;
      nvars = conshdlrdata->allconsanddatas[c]->nvars;
      assert(vars != NULL || nvars == 0);
         
      /* get transformed variables */
      SCIP_CALL( SCIPgetTransformedVars(scip, nvars, vars, vars) );

      /* resort variables in transformed problem, because the order might change while tranforming */
      SCIPsortPtr((void**)vars, SCIPvarComp, nvars);
      
#if 1
      /* do not capture after restart */
      if( SCIPgetNRuns(scip) == 1 )
      {
         /* capture all variables */
         for( v = nvars - 1; v >= 0; --v )
         {
            SCIP_CALL( SCIPcaptureVar(scip, vars[v]) );
         }
      }
#endif

      andcons = conshdlrdata->allconsanddatas[c]->cons;
      assert(andcons != NULL);

      /* in a restart the constraints might already be transformed */
      if( !SCIPconsIsTransformed(andcons) )
      {
         SCIP_CONS* transcons;
         
         SCIP_CALL( SCIPgetTransformedCons(scip, andcons, &transcons) );
         assert( transcons != NULL );
         conshdlrdata->allconsanddatas[c]->cons = transcons;
         
         resultant = SCIPgetResultantAnd(scip, transcons);
         /* insert new mapping */
         assert(!SCIPhashmapExists(conshdlrdata->hashmap, (void*)resultant));
         SCIP_CALL( SCIPhashmapInsert(conshdlrdata->hashmap, (void*)resultant, (void*)(conshdlrdata->allconsanddatas[c])) );

         /* capture constraint */
         SCIP_CALL( SCIPcaptureCons(scip, conshdlrdata->allconsanddatas[c]->cons) );
      }
      resultant = SCIPgetResultantAnd(scip, conshdlrdata->allconsanddatas[c]->cons);
      assert(SCIPhashmapExists(conshdlrdata->hashmap, (void*)resultant));
   }

   /* check each constraint and get transformed constraints */
   for( c = nconss - 1; c >= 0; --c )
   {
      SCIP_CONSDATA* consdata;
      SCIP_CONS* cons;
      int a;
      SCIP_Real rhs;
      SCIP_Real lhs;

      assert(conss != NULL);

      cons = conss[c];
      assert(cons != NULL);
      assert(SCIPconsIsTransformed(cons));

      consdata = SCIPconsGetData(cons);
      assert(consdata != NULL);

      /* if not happend already, get transformed linear constraint */
      assert(consdata->lincons != NULL);
      assert(consdata->linconstype > SCIP_INVALIDCONS);

      /* in a restart the linear constraint might already be transformed */
      if( !SCIPconsIsTransformed(consdata->lincons) )
      {
         SCIP_CONS* transcons;

         SCIP_CALL( SCIPgetTransformedCons(scip, consdata->lincons, &transcons) );
         assert( transcons != NULL );
         SCIP_CALL( SCIPcaptureCons(scip, transcons) );
         consdata->lincons = transcons;
      }

      lhs = consdata->lhs;
      rhs = consdata->rhs;

      assert(consdata->nconsanddatas == 0 || (consdata->consanddatas != NULL && consdata->andcoefs != NULL));
      for( a = consdata->nconsanddatas - 1; a >= 0; --a )
      {
         /* add rounding locks due to old variables in consanddata object */
         SCIP_CALL( lockRoundingAndCons(scip, cons, consdata->consanddatas[a], consdata->andcoefs[a], lhs, rhs) );
      }
   }
#else
#if 0 // locks from original variable are copied to transformed variables, so we don't need to add locks
   /* check each constraint and get transformed constraints */
   for( c = nconss - 1; c >= 0; --c )
   {
      SCIP_CONSDATA* consdata;
      SCIP_CONS* cons;
      int a;
      SCIP_Real rhs;
      SCIP_Real lhs;

      assert(conss != NULL);

      cons = conss[c];
      assert(cons != NULL);
      assert(SCIPconsIsTransformed(cons));

      consdata = SCIPconsGetData(cons);
      assert(consdata != NULL);

      lhs = consdata->lhs;
      rhs = consdata->rhs;


      assert(consdata->nconsanddatas == 0 || (consdata->consanddatas != NULL && consdata->andcoefs != NULL));
      for( a = consdata->nconsanddatas - 1; a >= 0; --a )
      {
         /* add rounding locks due to old variables in consanddata object */
         SCIP_CALL( lockRoundingAndCons(scip, cons, consdata->consanddatas[a], consdata->andcoefs[a], lhs, rhs) );
      }
   }
#endif
#endif

   /* decompose all pseudo boolean constraints into a "linear" constraint and "and" constraints */
   if( conshdlrdata->decomposeindicatorpbcons || conshdlrdata->decomposenormalpbcons )
   {
      for( c = 0; c < nconss; ++c )
      {
         SCIP_CONS* cons;
         SCIP_CONSDATA* consdata;
         SCIP_VAR** vars;
         SCIP_Real* coefs;
         int nvars;
 
         cons = conss[c];
         assert(cons != NULL);

         consdata = SCIPconsGetData(cons);
         assert(consdata != NULL);

         /* gets number of variables in linear constraint */
         SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );
         
         /* allocate temporary memory */
         SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
         SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
         
         /* get variables and coefficient of linear constraint */
         SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
         assert(nvars == 0 || (vars != NULL && coefs != NULL));

         if( consdata->issoftcons && conshdlrdata->decomposeindicatorpbcons )
         {
            SCIP_VAR* negindvar;
            char name[SCIP_MAXSTRLEN];
            SCIP_Real lhs;
            SCIP_Real rhs;
            SCIP_Bool initial;
            SCIP_Bool updateandconss;
            int v;
#if USEINDICATOR == FALSE
            SCIP_CONS* lincons;
            SCIP_Real maxact;
            SCIP_Real minact;
            SCIP_Real lb;
            SCIP_Real ub;
#else
            SCIP_CONS* indcons;
#endif         

            assert(consdata->weight != 0);
            assert(consdata->indvar != NULL);

            /* if it is a soft constraint, there should be no integer variable */
            assert(consdata->intvar == NULL);

            /* get negation of indicator variable */
            SCIP_CALL( SCIPgetNegatedVar(scip, consdata->indvar, &negindvar) );
            assert(negindvar != NULL);

            /* get sides of linear constraint */
            SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &lhs, &rhs) );
            assert(!SCIPisInfinity(scip, lhs));
            assert(!SCIPisInfinity(scip, -rhs));
            assert(SCIPisLE(scip, lhs, rhs));

            updateandconss = FALSE;

#if USEINDICATOR == FALSE
            maxact = 0.0;
            minact = 0.0;
            
            /* adding all linear coefficients up */
            for( v = nvars - 1; v >= 0; --v )
               if( coefs[v] > 0 )
                  maxact += coefs[v];
               else
                  minact += coefs[v];

            if( SCIPisInfinity(scip, maxact) )
            {
               SCIPwarningMessage("maxactivity = %g exceed infinity value.\n", maxact);
            }
            if( SCIPisInfinity(scip, -minact) )
            {
               SCIPwarningMessage("minactivity = %g exceed -infinity value.\n", minact);
            }

            /* @todo check whether it's better to set the initial flag to false */         
            initial = SCIPconsIsInitial(cons); //FALSE;

            /* first soft constraints for lhs */
            if( !SCIPisInfinity(scip, -lhs) )
            {
               /* first we are modelling the feasibility of the soft contraint by adding a slack variable */
               /* we ensure that if indvar == 1 => (a^T*x + ub*indvar >= lhs) */
               ub = lhs - minact;

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_lhs_part1", SCIPconsGetName(cons));
               
               SCIP_CALL( SCIPcreateConsLinear(scip, &lincons, name, nvars, vars, coefs, lhs, SCIPinfinity(scip),
                     initial, SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons),
                     SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons),
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               /* update and constraint flags */
               SCIP_CALL( updateAndConss(scip, cons) );
               updateandconss = TRUE;
            
               /* add artificial indicator variable */
               SCIP_CALL( SCIPaddCoefLinear(scip, lincons, consdata->indvar, ub) );
      
               SCIP_CALL( SCIPaddCons(scip, lincons) );
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, lincons, NULL) ) );
               SCIP_CALL( SCIPreleaseCons(scip, &lincons) );
      
               /* second we are modelling the implication that if the slack variable is on( negation is off), the constraint
                * is disabled, so only the cost arise if the slack variable is necessary */
               /* indvar == 1 => (a^T*x (+ ub * negindvar) <= lhs - 1) */
               ub = lhs - maxact - 1;
      
               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_lhs_part2", SCIPconsGetName(cons));

               SCIP_CALL( SCIPcreateConsLinear(scip, &lincons, name, nvars, vars, coefs, -SCIPinfinity(scip), lhs - 1,
                     initial, SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons),
                     SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons),
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );
               
               /* add artificial indicator variable */
               SCIP_CALL( SCIPaddCoefLinear(scip, lincons, negindvar, ub) );
            
               SCIP_CALL( SCIPaddCons(scip, lincons) );
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, lincons, NULL) ) );
               SCIP_CALL( SCIPreleaseCons(scip, &lincons) );
            }
   
            /* second soft constraints for rhs */
            if( !SCIPisInfinity(scip, rhs) )
            {
               /* first we are modelling the feasibility of the soft-constraint by adding a slack variable */
               /* indvar == 1 => (a^T*x + lb * indvar <= rhs) */
               lb = rhs - maxact;
               
               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_rhs_part1", SCIPconsGetName(cons));

               SCIP_CALL( SCIPcreateConsLinear(scip, &lincons, name, nvars, vars, coefs, -SCIPinfinity(scip), rhs,
                     initial, SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons),
                     SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons),
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               if( !updateandconss )
               {
                  /* update and constraint flags */
                  SCIP_CALL( updateAndConss(scip, cons) );
               }
               
               /* add artificial indicator variable */
               SCIP_CALL( SCIPaddCoefLinear(scip, lincons, consdata->indvar, lb) );
      
               SCIP_CALL( SCIPaddCons(scip, lincons) );
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, lincons, NULL) ) );
               SCIP_CALL( SCIPreleaseCons(scip, &lincons) );
            
               /* second we are modelling the implication that if the slack variable is on( negation is off), the constraint
                * is disabled, so only the cost arise if the slack variable is necessary */
               /* indvar == 1 => (a^T*x (+ lb * negindvar) >= rhs + 1) */
               lb = rhs - minact + 1;
      
               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_rhs_part2", SCIPconsGetName(cons));

               SCIP_CALL( SCIPcreateConsLinear(scip, &lincons, name, nvars, vars, coefs, rhs + 1, SCIPinfinity(scip),
                     initial, SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons),
                     SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons), SCIPconsIsModifiable(cons),
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               /* add artificial indicator variable */
               SCIP_CALL( SCIPaddCoefLinear(scip, lincons, negindvar, lb) );
      
               SCIP_CALL( SCIPaddCons(scip, lincons) );
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, lincons, NULL) ) );
               SCIP_CALL( SCIPreleaseCons(scip, &lincons) );
            }
#else /* with indicator */
            /* @todo check whether it's better to set the initial flag to false */         
            initial = SCIPconsIsInitial(cons); //FALSE;
            
            if( !SCIPisInfinity(scip, rhs) )
            {
               /* first we are modelling the implication that if the negation of the indicator variable is on, the constraint
                * is enabled */
               /* indvar == 0 => a^T*x <= rhs */

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_rhs_ind", SCIPconsGetName(cons));

               SCIP_CALL( SCIPcreateConsIndicator(scip, &indcons, name, negindvar, nvars, vars, coefs, rhs,
                     initial, SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons),
                     SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons), 
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               /* update and constraint flags */
               SCIP_CALL( updateAndConss(scip, cons) );
               updateandconss = TRUE;

               SCIP_CALL( SCIPaddCons(scip, indcons) );
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, indcons, NULL) ) );
               SCIP_CALL( SCIPreleaseCons(scip, &indcons) );
            }
   
            if( !SCIPisInfinity(scip, -lhs) )
            {
               /* second we are modelling the implication that if the negation of the indicator variable is on, the constraint
                * is enabled */
               /* change the a^T*x >= lhs to -a^Tx<= -lhs, for indicator constraint */

               for( v = nvars - 1; v >= 0; --v )
                  coefs[v] *= -1;

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_lhs_ind", SCIPconsGetName(cons));

               SCIP_CALL( SCIPcreateConsIndicator(scip, &indcons, name, negindvar, nvars, vars, coefs, -lhs,
                     initial, SCIPconsIsSeparated(cons), SCIPconsIsEnforced(cons), SCIPconsIsChecked(cons),
                     SCIPconsIsPropagated(cons), SCIPconsIsLocal(cons),
                     SCIPconsIsDynamic(cons), SCIPconsIsRemovable(cons), SCIPconsIsStickingAtNode(cons)) );

               if( !updateandconss )
               {
                  /* update and constraint flags */
                  SCIP_CALL( updateAndConss(scip, cons) );
               }

               SCIP_CALL( SCIPaddCons(scip, indcons) );
               SCIPdebug( SCIP_CALL( SCIPprintCons(scip, indcons, NULL) ) );
               SCIP_CALL( SCIPreleaseCons(scip, &indcons) );
            }
#endif
            /* remove pseudo boolean and corresponding linear constraint, new linear constraints were created,
             * and-constraints still active
             */
            SCIP_CALL( SCIPdelCons(scip, consdata->lincons) );
            SCIP_CALL( SCIPdelCons(scip, cons) );
         }
         /* no soft constraint */
         else if( !consdata->issoftcons && conshdlrdata->decomposenormalpbcons )
         {
            if( consdata->linconstype == SCIP_LINEAR )
            {
               /* todo: maybe better create a new linear constraint and let scip do the upgrade */
               
               /* mark linear constraint not to be upgraded - otherwise we loose control over it */
               SCIP_CALL( SCIPsetUpgradeConsLinear(scip, consdata->lincons, TRUE) );
            }
         
            /* update and constraint flags */
            SCIP_CALL( updateAndConss(scip, cons) );

#if 0 // not implemented correctly 
            if( consdata->intvar != NULL )
            {
               /* add auxiliary integer variables to linear constraint */
               SCIP_CALL( SCIPaddCoefLinear(scip, lincons, consdata->intvar, -1.0) );
            }
#endif
            /* remove pseudo boolean constraint, old linear constraint is still active, and-constraints too */
            SCIP_CALL( SCIPdelCons(scip, cons) );
         }

         /* free temporary memory */
         SCIPfreeBufferArray(scip, &coefs);
         SCIPfreeBufferArray(scip, &vars);
      }
   }
   
   return SCIP_OKAY;
}


/** presolving deinitialization method of constraint handler (called after presolving has been finished) */
#if 0
static
SCIP_DECL_CONSEXITPRE(consExitprePseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consExitprePseudoboolean NULL
#endif


/** solving process initialization method of constraint handler (called when branch and bound process is about to begin) */
#if 0
static
SCIP_DECL_CONSINITSOL(consInitsolPseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consInitsolPseudoboolean NULL
#endif


/** solving process deinitialization method of constraint handler (called before branch and bound process data is freed) */
#if 0
static
SCIP_DECL_CONSEXITSOL(consExitsolPseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consExitsolPseudoboolean NULL
#endif


/** frees specific constraint data */
static
SCIP_DECL_CONSDELETE(consDeletePseudoboolean)
{  /*lint --e{715}*/

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(cons != NULL);
   assert(consdata != NULL);
   assert(*consdata != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

#if 0
   if( (*consdata)->nconsanddatas > 0 )
   {
      int c;

      assert((*consdata)->consanddatas != NULL);
      assert((*consdata)->andcoefs != NULL);

      for( c = (*consdata)->nconsanddatas - 1; c >= 0; --c )
      {
         CONSANDDATA* consanddata;

         consanddata = (*consdata)->consanddatas[c];
         assert(consanddata != NULL);
      
         if( consanddata->deleted )
            continue;

         SCIP_CALL( removeOldLocks(scip, cons, consanddata, (*consdata)->andcoefs[c], (*consdata)->lhs, (*consdata)->rhs) );
      }
   }
#endif


   /* free pseudo boolean constraint */
   SCIP_CALL( consdataFree(scip, consdata) );

   return SCIP_OKAY;
}

/** transforms constraint data into data belonging to the transformed problem */ 
static
SCIP_DECL_CONSTRANS(consTransPseudoboolean)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* sourcedata;
   SCIP_CONSDATA* targetdata;
   SCIP_CONS** andconss;
   int c;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(SCIPgetStage(scip) == SCIP_STAGE_TRANSFORMING);
   assert(sourcecons != NULL);
   assert(targetcons != NULL);

   sourcedata = SCIPconsGetData(sourcecons);
   assert(sourcedata != NULL);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   assert(sourcedata->nconsanddatas == 0 || sourcedata->consanddatas != NULL);

   /* allocate temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &andconss, sourcedata->nconsanddatas) );

   /* copy and-constraints */
   for( c = sourcedata->nconsanddatas - 1; c >= 0; --c )
   {
      assert(sourcedata->consanddatas[c] != NULL);
      andconss[c] = sourcedata->consanddatas[c]->cons;
      assert(andconss[c] != NULL);
   }

   /* create linear constraint data for target constraint */
   SCIP_CALL( consdataCreate(scip, conshdlr, &targetdata, sourcedata->lincons, sourcedata->linconstype, 
         andconss, sourcedata->andcoefs, sourcedata->nconsanddatas,     
         sourcedata->indvar, sourcedata->weight, sourcedata->issoftcons, sourcedata->intvar, sourcedata->lhs, sourcedata->rhs) );

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &andconss);

   /* create target constraint */
   SCIP_CALL( SCIPcreateCons(scip, targetcons, SCIPconsGetName(sourcecons), conshdlr, targetdata,
         SCIPconsIsInitial(sourcecons), SCIPconsIsSeparated(sourcecons), SCIPconsIsEnforced(sourcecons),
         SCIPconsIsChecked(sourcecons), SCIPconsIsPropagated(sourcecons),
         SCIPconsIsLocal(sourcecons), SCIPconsIsModifiable(sourcecons),
         SCIPconsIsDynamic(sourcecons), SCIPconsIsRemovable(sourcecons), SCIPconsIsStickingAtNode(sourcecons)) );

   return SCIP_OKAY;
}


/** LP initialization method of constraint handler */
#if 0
static
SCIP_DECL_CONSINITLP(consInitlpPseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consInitlpPseudoboolean NULL
#endif


/** separation method of constraint handler for LP solutions */
#if 0
static
SCIP_DECL_CONSSEPALP(consSepalpPseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consSepalpPseudoboolean NULL
#endif


/** separation method of constraint handler for arbitrary primal solutions */
#if 0
static
SCIP_DECL_CONSSEPASOL(consSepasolPseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consSepasolPseudoboolean NULL
#endif


/** constraint enforcing method of constraint handler for LP solutions */
#if 0
static
SCIP_DECL_CONSENFOLP(consEnfolpPseudoboolean)
{  /*lint --e{715}*/
   SCIP_Bool violated;
#if 0 /* now linear constraint does it itself */
   int c;
#endif

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(result != NULL);

   violated = FALSE;
   
#if 0 /* now linear constraint does it itself */
   /* check linear constraint of pseudoboolean constraint for feasibility */
   for( c = 0; c < nconss && !violated; ++c )
   {
      SCIP_CALL( checkCons(scip, conss[c], NULL, &violated) );
   }
#endif
   /* check all and-constraints */
   if( !violated )
   {
      SCIP_CALL( checkAndConss(scip, conshdlr, NULL, &violated) );
   }
   
   if( violated )
      *result = SCIP_INFEASIBLE;
   else
      *result = SCIP_FEASIBLE;
   
   return SCIP_OKAY;
}
#else
#define consEnfolpPseudoboolean NULL
#endif


/** constraint enforcing method of constraint handler for pseudo solutions */
#if 0
static
SCIP_DECL_CONSENFOPS(consEnfopsPseudoboolean)
{  /*lint --e{715}*/
   SCIP_Bool violated;
#if 0 /* now linear constraint does it itself */
   int c;
#endif   
   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(result != NULL);

   violated = FALSE;
   
#if 0 /* now linear constraint does it itself */
   /* check all pseudo boolean constraints for feasibility */
   for( c = 0; c < nconss && !violated; ++c )
   {
      SCIP_CALL( checkCons(scip, conss[c], NULL, &violated) );
   }
#endif

   /* check all and-constraints */
   if( !violated )
   {
      SCIP_CALL( checkAndConss(scip, conshdlr, NULL, &violated) );
   }

   if( violated )
      *result = SCIP_INFEASIBLE;
   else
      *result = SCIP_FEASIBLE;
   
   return SCIP_OKAY;
}
#else
#define consEnfopsPseudoboolean NULL
#endif


/** feasibility check method of constraint handler for integral solutions */
static
SCIP_DECL_CONSCHECK(consCheckPseudoboolean)
{  /*lint --e{715}*/
   SCIP_Bool violated;
   int c;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(sol != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(result != NULL);

   violated = FALSE;
   
#if 0 /* now linear constraint does it itself */
   /* check all pseudo boolean constraints for feasibility */
   for( c = 0; c < nconss && !violated; ++c )
   {
      SCIP_CALL( checkCons(scip, conss[c], sol, &violated) );
   }
#endif
   if( nconss > 0 )
   {
      if( SCIPconsIsOriginal(conss[0]) )
      {
         SCIP_CONSDATA* consdata;

         for( c = nconss - 1; c >= 0 && !violated; --c )
         {
            consdata = SCIPconsGetData(conss[c]);
            assert(consdata != NULL);

            if( consdata->issoftcons )
            {
               assert(consdata->indvar != NULL);
               if( SCIPisEQ(scip, SCIPgetSolVal(scip, sol, consdata->indvar), 1.0) )
                  continue;
            }

            SCIP_CALL( checkOrigPbCons(scip, conss[c], sol, &violated, printreason) );
         }
      }
      else
      {
         /* check all and-constraints */
         if( !violated )
         {
            SCIP_CALL( checkAndConss(scip, conshdlr, sol, &violated) );
         }
      }
   }

   if( violated )
      *result = SCIP_INFEASIBLE;
   else
      *result = SCIP_FEASIBLE;
   
   return SCIP_OKAY;
}


/** domain propagation method of constraint handler */
#if 0
static
SCIP_DECL_CONSPROP(consPropPseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consPropPseudoboolean NULL
#endif


/** presolving method of constraint handler */
static
SCIP_DECL_CONSPRESOL(consPresolPseudoboolean)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_Bool cutoff;
   int firstchange;
   int firstupgradetry;
   int oldnfixedvars;
   int oldnaggrvars;
   int oldnchgbds;
   int oldndelconss;
   int oldnupgdconss;
   int oldnchgcoefs;
   int oldnchgsides;
   int c;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(result != NULL);

   /* remember old preprocessing counters */
   oldnfixedvars = *nfixedvars;
   oldnaggrvars = *naggrvars;
   oldnchgbds = *nchgbds;
   oldndelconss = *ndelconss;
   oldnupgdconss = *nupgdconss;
   oldnchgcoefs = *nchgcoefs;
   oldnchgsides = *nchgsides;

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);

   /* compute all changes in consanddata objects */
   SCIP_CALL( computeConsAndDataChanges(scip, conshdlrdata) );

   firstchange = INT_MAX;
   firstupgradetry = INT_MAX;
   cutoff = FALSE;

   for( c = 0; c < nconss && !cutoff && !SCIPisStopped(scip); ++c )
   {
      SCIP_CONS* cons;
      SCIP_CONSDATA* consdata;
      SCIP_VAR** vars;
      SCIP_Real* coefs;
      int nvars;
      SCIP_VAR** linvars;
      SCIP_Real* lincoefs;
      int nlinvars;
      SCIP_VAR** andress;
      SCIP_Real* andcoefs;
      int nandress;
      SCIP_Real newlhs;
      SCIP_Real newrhs;

      cons = conss[c];
      assert(cons != NULL);
      assert(SCIPconsIsActive(cons));

      consdata = SCIPconsGetData(cons);
      assert(consdata != NULL);
      assert(consdata->lincons != NULL);

      /* if linear constraint is redundant, than pseudoboolean constraint is redundant too */
      if( SCIPconsIsDeleted(consdata->lincons) )
      {
         /* update and constraint flags */
         SCIP_CALL( updateAndConss(scip, cons) );

         SCIP_CALL( SCIPdelCons(scip, cons) );
         ++(*ndelconss);
         continue;
      }

      /* get sides of linear constraint */
      SCIP_CALL( getLinearConsSides(scip, consdata->lincons, consdata->linconstype, &newlhs, &newrhs) );
      assert(!SCIPisInfinity(scip, newlhs));
      assert(!SCIPisInfinity(scip, -newrhs));
      assert(SCIPisLE(scip, newlhs, newrhs));

      /* gets number of variables in linear constraint */
      SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );
      
      /* allocate temporary memory */
      SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &linvars, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &lincoefs, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &andress, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nvars) );
      
      /* get variables and coefficient of linear constraint */
      SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
      assert(nvars == 0 || (vars != NULL && coefs != NULL));
      
      /* calculate all not artificial linear variables and all artificial and-resultants which will be ordered like the
       * 'consanddatas' such that the and-resultant of the and-constraint is the and-resultant in the 'andress' array
       * afterwards 
       */
      SCIP_CALL( getLinVarsAndAndRess(scip, cons, vars, coefs, nvars, linvars, lincoefs, &nlinvars, andress, andcoefs, &nandress) );

      /* update all locks inside this constraint and all captures on all and-constraints */
      SCIP_CALL( correctLocksAndCaptures(scip, cons, conshdlrdata, newlhs, newrhs, andress, andcoefs, nandress) );

      /* we can only presolve pseudoboolean constraints, that are not modifiable */
      if( SCIPconsIsModifiable(cons) )
         goto CONTTERMINATE;

      SCIPdebugMessage("presolving pseudoboolean constraint <%s>\n", SCIPconsGetName(cons));
      SCIPdebug(SCIP_CALL( SCIPprintCons(scip, cons, NULL) ));

      /* remember the first changed constraint to begin the next aggregation round with */
      if( firstchange == INT_MAX && consdata->changed )
         firstchange = c;

      if( consdata->changed )
      {
         /* try upgrading pseudoboolean constraint to a linear constraint and/or remove possible and-constraints */
         SCIP_CALL( tryUpgrading(scip, cons, conshdlrdata, ndelconss, nfixedvars, nchgcoefs, nchgsides, &cutoff) );
         if( cutoff )
            goto CONTTERMINATE;
      }  

      /* if upgrading deleted the pseudoboolean constraint we go on */
      if( !SCIPconsIsActive(cons) )
         goto CONTTERMINATE;

      /* remember the first constraint that was not yet tried to be upgraded, to begin the next upgrading round with */
      if( firstupgradetry == INT_MAX && !consdata->upgradetried )
         firstupgradetry = c;

      while( !consdata->presolved && !SCIPisStopped(scip) )
      {
         /* mark constraint being presolved and propagated */
         consdata->presolved = TRUE;
         consdata->propagated = TRUE;

         /* add cliques to the clique table */
         SCIP_CALL( addCliques(scip, cons, &cutoff, naggrvars, nchgbds) );
         if( cutoff )
            break;
         
         /* propagate constraint */
         SCIP_CALL( propagateCons(scip, cons, &cutoff, ndelconss) );
         if( cutoff )
            break;
      }

   CONTTERMINATE:
      /* free temporary memory */
      SCIPfreeBufferArray(scip, &andcoefs);
      SCIPfreeBufferArray(scip, &andress);
      SCIPfreeBufferArray(scip, &lincoefs);
      SCIPfreeBufferArray(scip, &linvars);
      SCIPfreeBufferArray(scip, &coefs);
      SCIPfreeBufferArray(scip, &vars);
   }

   /* delete unused information in constraint handler data */
   SCIP_CALL( correctConshdlrdata(scip, conshdlrdata, ndelconss) );

   /* return the correct result code */
   if( cutoff )
      *result = SCIP_CUTOFF;
   else if( *nfixedvars > oldnfixedvars || *naggrvars > oldnaggrvars || *nchgbds > oldnchgbds || *ndelconss > oldndelconss
      || *nupgdconss > oldnupgdconss || *nchgcoefs > oldnchgcoefs || *nchgsides > oldnchgsides )
      *result = SCIP_SUCCESS;
   else
      *result = SCIP_DIDNOTFIND;

   return SCIP_OKAY;
}


/** propagation conflict resolving method of constraint handler */
#if 0
static
SCIP_DECL_CONSRESPROP(consRespropPseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consRespropPseudoboolean NULL
#endif


/** variable rounding lock method of constraint handler */
static
SCIP_DECL_CONSLOCK(consLockPseudoboolean)
{  /*lint --e{715}*/
   SCIP_CONSDATA* consdata;
   SCIP_Real lhs;
   SCIP_Real rhs;
   SCIP_Bool haslhs;
   SCIP_Bool hasrhs;
   int v;
   int c;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   lhs = consdata->lhs;
   rhs = consdata->rhs;
   assert(!SCIPisInfinity(scip, lhs));
   assert(!SCIPisInfinity(scip, -rhs));
   assert(SCIPisLE(scip, lhs, rhs));

   haslhs = !SCIPisInfinity(scip, -lhs);
   hasrhs = !SCIPisInfinity(scip, rhs);

   SCIPdebugMessage("%socking constraint <%s> by [%d;%d].\n", (nlocksneg < 0) || (nlockspos < 0) ? "Unl" : "L", SCIPconsGetName(cons), nlocksneg, nlockspos);

   /* update rounding locks of every single variable corresponding to the and-constraints */
   for( c = consdata->nconsanddatas - 1; c >= 0; --c )
   {
      SCIP_VAR* andres;
      SCIP_VAR** andvars;
      SCIP_Real val;
      int nandvars;
      SCIP_CONS* andcons;
      CONSANDDATA* consanddata;

      consanddata = consdata->consanddatas[c];

      if( consanddata->deleted )
         continue;

      assert(consanddata != NULL);
      andcons = consanddata->cons;

      /* in stage SCIP_STAGE_FREETRANS the captures of all and-constraints are already removed (in CONSEXIT), so all
       * and-constraint pointers should be NULL
       *
       * NOTE: because of don not having any constraints anymore we cannot delete the locks in stage
       * SCIP_STAGE_FREETRANS, and we cannnot do it in CONSEXIT either because there we are in stage
       * SCIP_STAGE_TRANSFORMED where we cannot call SCIPaddVarLocks
       */
      assert((SCIPgetStage(scip) == SCIP_STAGE_FREETRANS) == (andcons == NULL));

      if( andcons == NULL )
      {
         /* we should have no new variables */
         assert(consanddata->nnewvars == 0);
         
         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->vars), consanddata->svars);
         SCIPfreeBlockMemoryArrayNull(scip, &(consanddata->newvars), consanddata->snewvars);

         consanddata->nvars = 0;
         consanddata->svars = 0;
         consanddata->nnewvars = 0;
         consanddata->snewvars = 0;
         consanddata->deleted = TRUE;

         continue;
      }
      assert(andcons != NULL);
      if( consanddata->nnewvars > 0 )
      {
         andvars = consanddata->newvars;
         nandvars = consanddata->nnewvars;
      }
      else
      {
         andvars = consanddata->vars;
         nandvars = consanddata->nvars;
      }

      /* probably we need to store the resultant too, now it's not possible to remove the resultant from the and-constraint */
      andres = SCIPgetResultantAnd(scip, andcons);
      assert(nandvars == 0 || andvars != NULL);
      assert(andres != NULL);
      val = consdata->andcoefs[c];

      /* lock variables */
      if( SCIPisPositive(scip, val) )
      {
         if( haslhs )
         {
            for( v = nandvars - 1; v >= 0; --v )
            {
               //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), nlockspos, nlocksneg);
               SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], nlockspos, nlocksneg) );
            }
            //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), nlocksneg + nlockspos, nlocksneg + nlockspos);
            SCIP_CALL( SCIPaddVarLocks(scip, andres, nlocksneg + nlockspos, nlocksneg + nlockspos) );
         }
         if( hasrhs )
         {
            for( v = nandvars - 1; v >= 0; --v )
            {
               //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), nlockspos, nlocksneg);
               SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], nlocksneg, nlockspos) );
            }
            /* don't double the locks on the and-resultant */
            if( !haslhs )
            {
               //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), nlocksneg + nlockspos, nlocksneg + nlockspos);
               SCIP_CALL( SCIPaddVarLocks(scip, andres, nlocksneg + nlockspos, nlocksneg + nlockspos) );
            }
         }
      }
      else
      {
         if( haslhs )
         {
            for( v = nandvars - 1; v >= 0; --v )
            {
               //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), nlockspos, nlocksneg);
               SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], nlocksneg, nlockspos) );
            }
            //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), nlocksneg + nlockspos, nlocksneg + nlockspos);
            SCIP_CALL( SCIPaddVarLocks(scip, andres, nlocksneg + nlockspos, nlocksneg + nlockspos) );
         }
         if( hasrhs )
         {
            for( v = nandvars - 1; v >= 0; --v )
            {
               //printf("locking var <%s>, [%d,%d] \n", SCIPvarGetName(andvars[v]), nlockspos, nlocksneg);
               SCIP_CALL( SCIPaddVarLocks(scip, andvars[v], nlockspos, nlocksneg) );
            }
            /* don't double the locks on the and-resultant */
            if( !haslhs )
            {
               //printf("locking resvar <%s>, [%d,%d] \n", SCIPvarGetName(andres), nlocksneg + nlockspos, nlocksneg + nlockspos);
               SCIP_CALL( SCIPaddVarLocks(scip, andres, nlocksneg + nlockspos, nlocksneg + nlockspos) );
            }
         }
      }
   }

   return SCIP_OKAY;
}


/** constraint activation notification method of constraint handler */
#if 0
static
SCIP_DECL_CONSACTIVE(consActivePseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consActivePseudoboolean NULL
#endif


/** constraint deactivation notification method of constraint handler */
#if 0
static
SCIP_DECL_CONSDEACTIVE(consDeactivePseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consDeactivePseudoboolean NULL
#endif


/** constraint enabling notification method of constraint handler */
#if 0
static
SCIP_DECL_CONSENABLE(consEnablePseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consEnablePseudoboolean NULL
#endif


/** constraint disabling notification method of constraint handler */
#if 0
static
SCIP_DECL_CONSDISABLE(consDisablePseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consDisablePseudoboolean NULL
#endif


/** constraint display method of constraint handler */
static
SCIP_DECL_CONSPRINT(consPrintPseudoboolean)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(cons != NULL);

   SCIP_CALL( consdataPrint(scip, cons, file) );

   return SCIP_OKAY;
}


/** constraint copying method of constraint handler */
static
SCIP_DECL_CONSCOPY(consCopyPseudoboolean)
{  /*lint --e{715}*/
   const char* consname;

   assert(scip != NULL);
   assert(sourcescip != NULL);
   assert(sourcecons != NULL);

   if( name != NULL )
      consname = name;
   else
      consname = SCIPconsGetName(sourcecons);

   SCIP_CALL( copyConsPseudoboolean(scip, cons, sourcescip, sourcecons, consname, varmap, consmap, 
         initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode, global, valid) );
   assert(cons != NULL || *valid == FALSE);

   return SCIP_OKAY;
}

/** constraint parsing method of constraint handler */
#if 0
static
SCIP_DECL_CONSPARSE(consParsePseudoboolean)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of pseudoboolean constraint handler not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define consParsePseudoboolean NULL
#endif


/*
 * constraint specific interface methods
 */

/** creates the handler for pseudoboolean constraints and includes it in SCIP */
SCIP_RETCODE SCIPincludeConshdlrPseudoboolean(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   /* create pseudoboolean constraint handler data */
   SCIP_CALL( conshdlrdataCreate(scip, &conshdlrdata) );

   /* include constraint handler */
   SCIP_CALL( SCIPincludeConshdlr(scip, CONSHDLR_NAME, CONSHDLR_DESC,
         CONSHDLR_SEPAPRIORITY, CONSHDLR_ENFOPRIORITY, CONSHDLR_CHECKPRIORITY,
         CONSHDLR_SEPAFREQ, CONSHDLR_PROPFREQ, CONSHDLR_EAGERFREQ, CONSHDLR_MAXPREROUNDS, 
         CONSHDLR_DELAYSEPA, CONSHDLR_DELAYPROP, CONSHDLR_DELAYPRESOL, CONSHDLR_NEEDSCONS,
         conshdlrCopyPseudoboolean,
         consFreePseudoboolean, consInitPseudoboolean, consExitPseudoboolean, 
         consInitprePseudoboolean, consExitprePseudoboolean, consInitsolPseudoboolean, consExitsolPseudoboolean,
         consDeletePseudoboolean, consTransPseudoboolean, consInitlpPseudoboolean,
         consSepalpPseudoboolean, consSepasolPseudoboolean, consEnfolpPseudoboolean, consEnfopsPseudoboolean, consCheckPseudoboolean, 
         consPropPseudoboolean, consPresolPseudoboolean, consRespropPseudoboolean, consLockPseudoboolean,
         consActivePseudoboolean, consDeactivePseudoboolean, 
         consEnablePseudoboolean, consDisablePseudoboolean,
         consPrintPseudoboolean, consCopyPseudoboolean, consParsePseudoboolean,
         conshdlrdata) );

   /* add pseudoboolean constraint handler parameters */
   SCIP_CALL( SCIPaddBoolParam(scip,
         "constraints/"CONSHDLR_NAME"/decomposenormal",
         "decompose all normal pseudo boolean constraint into a \"linear\" constraint \"and\" constraints",
         &conshdlrdata->decomposenormalpbcons, TRUE, DEFAULT_DECOMPOSENORMALPBCONS, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "constraints/"CONSHDLR_NAME"/decomposeindicator",
         "decompose all indicator pseudo boolean constraint into a \"linear\" constraint \"and\" constraints",
         &conshdlrdata->decomposeindicatorpbcons, TRUE, DEFAULT_DECOMPOSEINDICATORPBCONS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
         "constraints/"CONSHDLR_NAME"/nlcseparate", "should the nonlinear constraints be separated during LP processing?",
         NULL, TRUE, DEFAULT_SEPARATENONLINEAR, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "constraints/"CONSHDLR_NAME"/nlcpropagate", "should the nonlinear constraints be propagated during node processing?",
         NULL, TRUE, DEFAULT_PROPAGATENONLINEAR, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "constraints/"CONSHDLR_NAME"/nlcremovable", "should the nonlinear constraints be removable?",
         NULL, TRUE, DEFAULT_REMOVABLENONLINEAR, NULL, NULL) );

   return SCIP_OKAY;
}

/** creates and captures a pseudoboolean constraint, with given linear and and-constraints */
SCIP_RETCODE SCIPcreateConsPseudobooleanWithConss(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   SCIP_CONS*            lincons,            /**< associated linear constraint */
   SCIP_LINEARCONSTYPE   linconstype,        /**< linear constraint type of associated linear constraint */
   SCIP_CONS**           andconss,           /**< associated and-constraints */
   SCIP_Real*            andcoefs,           /**< associated coefficients of and-constraints */
   int                   nandconss,          /**< number of associated and-constraints */
   SCIP_VAR*             indvar,             /**< indicator variable if it's a soft constraint, or NULL */
   SCIP_Real             weight,             /**< weight of the soft constraint, if it is one */
   SCIP_Bool             issoftcons,         /**< is this a soft constraint */
   SCIP_VAR*             intvar,             /**< a artificial variable which was added only for the objective function,
                                              *   if this variable is not NULL this constraint (without this integer
                                              *   variable) describes the objective funktion */
   SCIP_Real             lhs,                /**< left hand side of constraint */
   SCIP_Real             rhs,                /**< right hand side of constraint */
   SCIP_Bool             initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                              *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool             separate,           /**< should the constraint be separated during LP processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             check,              /**< should the constraint be checked for feasibility?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             propagate,          /**< should the constraint be propagated during node processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             local,              /**< is constraint only valid locally?
                                              *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool             dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which 
                                              *   are seperated as constraints. */
   SCIP_Bool             removable,          /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   SCIP_Bool             stickingatnode      /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   )
{
   CONSANDDATA* newdata;
   CONSANDDATA* tmpdata;
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   SCIP_VAR* res;
   int nvars;
   SCIP_Bool memisinvalid;
   int c;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(lincons != NULL);
   assert(linconstype > SCIP_INVALIDCONS);
   assert(andconss != NULL);
   assert(andcoefs != NULL);
   assert(nandconss >= 1);
   assert(issoftcons == (indvar != NULL));

   /* find the pseudoboolean constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   if( conshdlr == NULL )
   {
      SCIPerrorMessage("pseudo boolean constraint handler not found\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->hashmap != NULL);
   assert(conshdlrdata->hashtable != NULL);
   assert(conshdlrdata->allconsanddatas != NULL);
   assert(conshdlrdata->nallconsanddatas <= conshdlrdata->sallconsanddatas);

   memisinvalid = TRUE;
   newdata = NULL;

   /* create hash map and hash table entries */
   for( c = nandconss - 1; c >= 0; --c )
   {
      assert(andconss[c] != NULL);
      res = SCIPgetResultantAnd(scip, andconss[c]);
      vars = SCIPgetVarsAnd(scip, andconss[c]);
      nvars = SCIPgetNVarsAnd(scip, andconss[c]);
      assert(vars != NULL && nvars > 0);
      assert(res != NULL);

      /* if allocated memory in this for loop was already used, allocate a new block, otherwise we only need to copy the variables */
      if( memisinvalid )
      {
         /* allocate memory for a possible new consanddata object */
         SCIP_CALL( SCIPallocBlockMemory(scip, &newdata) );
         SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(newdata->vars), vars, nvars) );
         newdata->svars = nvars;
         newdata->newvars = NULL;
         newdata->nnewvars = 0;
         newdata->snewvars = 0;
         newdata->deleted = FALSE;
         newdata->nuses = 0;
         newdata->origcons = NULL;
      }
      else
      {
         assert(newdata != NULL);
         /* resize variable array if necessary */
         if( newdata->svars < nvars )
         {
            SCIP_CALL( SCIPensureBlockMemoryArray(scip, &(newdata->vars), &(newdata->svars), nvars) );
         }

         /* copy variables in already allocated array */
         BMScopyMemoryArray(newdata->vars, vars, nvars);
      }
      
      /* sort variables */
      SCIPsortPtr((void**)(newdata->vars), SCIPvarComp, nvars);

      newdata->nvars = nvars;
      assert(newdata->vars != NULL && newdata->nvars > 0);

      newdata->cons = andconss[c];
      
      /* get constraint from current hash table with same variables as andconss[c] */
      tmpdata = (CONSANDDATA*)(SCIPhashtableRetrieve(conshdlrdata->hashtable, (void*)newdata));
      assert(tmpdata == NULL || tmpdata->cons != NULL);

      if( tmpdata == NULL || tmpdata->cons != andconss[c] )
      {
         if( tmpdata != NULL && tmpdata->cons != NULL )
         {
            SCIPwarningMessage("Another and-constraint with the same vaiables but different and-resultant is added to the global and-constraint hashtable of pseudoboolean constraint handler.\n");
         }

         /* resize data for all and-constraints if necessary */
         if( conshdlrdata->nallconsanddatas == conshdlrdata->sallconsanddatas )
         {
            SCIP_CALL( SCIPensureBlockMemoryArray(scip, &(conshdlrdata->allconsanddatas), &(conshdlrdata->sallconsanddatas), SCIPcalcMemGrowSize(scip, conshdlrdata->sallconsanddatas + 1)) );
         }
         
         conshdlrdata->allconsanddatas[conshdlrdata->nallconsanddatas] = newdata;
         ++(conshdlrdata->nallconsanddatas);

         /* increase usage of data object */
         ++(newdata->nuses);

         /* no such and-constraint in current hash table: insert the new object into hash table */  
         SCIP_CALL( SCIPhashtableInsert(conshdlrdata->hashtable, (void*)newdata) );
         
         /* if newdata object was new we want to allocate new memory in next loop iteration */
         memisinvalid = TRUE;
         assert(!SCIPhashmapExists(conshdlrdata->hashmap, (void*)res));
         
         /* capture and-constraint */
         SCIP_CALL( SCIPcaptureCons(scip, newdata->cons) );

         /* insert new mapping */
         SCIP_CALL( SCIPhashmapInsert(conshdlrdata->hashmap, (void*)res, (void*)newdata) );
      }
      else
      {
         assert(SCIPhashmapExists(conshdlrdata->hashmap, (void*)res));
         memisinvalid = FALSE;
         
         /* increase usage of data object */
         ++(tmpdata->nuses);
      }
   }

   if( !memisinvalid )
   {
      /* free temporary memory */
      SCIPfreeBlockMemoryArray(scip, &(newdata->vars), newdata->svars);
      SCIPfreeBlockMemory(scip, &newdata);
   }
  
   /* adjust right hand side */
   if( SCIPisInfinity(scip, rhs) )
      rhs = SCIPinfinity(scip);
   else if( SCIPisInfinity(scip, -rhs) )
      rhs = -SCIPinfinity(scip);

   /* capture linear constraint */
   SCIP_CALL( SCIPcaptureCons(scip, lincons) );

   if( linconstype == SCIP_LINEAR )
   {
      /* todo: make the constraint upgrade flag global, now it works only for the common linear constraint */
      /* mark linear constraint not to be upgraded - otherwise we loose control over it */
      SCIP_CALL( SCIPmarkDoNotUpgradeConsLinear(scip, lincons) );
   }

   /* create constraint data */
   /* checking for and-constraints will be FALSE, we check all information in this constraint handler */
   SCIP_CALL( consdataCreate(scip, conshdlr, &consdata, lincons, linconstype, andconss, andcoefs, nandconss, 
         indvar, weight, issoftcons, intvar, lhs, rhs) );
   assert(consdata != NULL);
   
   /* create constraint */
   SCIP_CALL( SCIPcreateCons(scip, cons, name, conshdlr, consdata, initial, separate, enforce, check, propagate,
         local, modifiable, dynamic, removable, stickingatnode) );
   
   return SCIP_OKAY;
}

/** creates and captures a pseudoboolean constraint */
SCIP_RETCODE SCIPcreateConsPseudoboolean(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   SCIP_VAR**            linvars,            /**< variables of the linear part, or NULL */
   int                   nlinvars,           /**< number of variables of the linear part */
   SCIP_Real*            linvals,            /**< coefficients of linear part, or NULL */
   SCIP_VAR***           terms,              /**< nonlinear terms of variables, or NULL */
   int                   nterms,             /**< number of terms of variables of nonlinear term */
   int*                  ntermvars,          /**< number of variables in nonlinear terms, or NULL */
   SCIP_Real*            termvals,           /**< coefficients of nonlinear parts, or NULL */
   SCIP_VAR*             indvar,             /**< indicator variable if it's a soft constraint, or NULL */
   SCIP_Real             weight,             /**< weight of the soft constraint, if it is one */
   SCIP_Bool             issoftcons,         /**< is this a soft constraint */
   SCIP_VAR*             intvar,             /**< a artificial variable which was added only for the objective function,
                                              *   if this variable is not NULL this constraint (without this integer
                                              *   variable) describes the objective funktion */
   SCIP_Real             lhs,                /**< left hand side of constraint */
   SCIP_Real             rhs,                /**< right hand side of constraint */
   SCIP_Bool             initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                              *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool             separate,           /**< should the constraint be separated during LP processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             check,              /**< should the constraint be checked for feasibility?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             propagate,          /**< should the constraint be propagated during node processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             local,              /**< is constraint only valid locally?
                                              *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool             dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which 
                                              *   are seperated as constraints. */
   SCIP_Bool             removable,          /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   SCIP_Bool             stickingatnode      /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSDATA* consdata;
   SCIP_VAR** andress;
   SCIP_CONS** andconss;
   SCIP_Real* andcoefs;
   int nandconss;
   SCIP_CONS* lincons;
   SCIP_LINEARCONSTYPE linconstype;
   int c;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(nlinvars == 0 || (linvars != NULL && linvals != NULL));
   assert(nterms == 0 || (terms != NULL && termvals != NULL && ntermvars != NULL));
   assert(issoftcons == (indvar != NULL));

   /* find the pseudoboolean constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   if( conshdlr == NULL )
   {
      SCIPerrorMessage("pseudo boolean constraint handler not found\n");
      return SCIP_PLUGINNOTFOUND;
   }

#if 0 //does not work
   /* if you try to read two instances after each other without solvuing the first, we did not clear the constraint
    * handler data, so do it when creating the first cnstraint 
    */
   if( SCIPconshdlrGetNConss(conshdlr) == 0 )
   {
      SCIP_CONSHDLRDATA* conshdlrdata;

      conshdlrdata = SCIPconshdlrGetData(conshdlr);
      assert(conshdlrdata != NULL);

      printf("clearing ...\n");
      /* clear constraint handler data */
      SCIP_CALL( conshdlrdataClear(scip, &conshdlrdata) );
   }
#endif

#if USEINDICATOR == TRUE
   if( issoftcons && modifiable )
   {
      SCIPerrorMessage("Indicator constraint handler can't work with modifiable constraints\n");
      return SCIP_INVALIDDATA;
   }
#endif

   /* get temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &andconss, nterms) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andress, nterms) );
   SCIP_CALL( SCIPallocBufferArray(scip, &andcoefs, nterms) );

   nandconss = 0;
   /* create and-constraints */
   SCIP_CALL( createAndAddAnds(scip, conshdlr, terms, termvals, nterms, ntermvars,  
         initial, enforce, check, local, modifiable, dynamic, stickingatnode, 
         andconss, andcoefs, &nandconss) );
   assert(nterms >= nandconss);

   /* get all and-resultants for linear constraint */
   for( c = nandconss - 1; c >= 0; --c )
   {
      assert(andconss[c] != NULL);
      andress[c] = SCIPgetResultantAnd(scip, andconss[c]);
   }

   linconstype = -1;

   /* adjust right hand side */
   if( SCIPisInfinity(scip, rhs) )
      rhs = SCIPinfinity(scip);
   else if( SCIPisInfinity(scip, -rhs) )
      rhs = -SCIPinfinity(scip);

   /* create and add linear constraint */
   /* checking for original linear constraint will be FALSE, tranformed linear constraints get the check flag like this
    * pseudoboolean constraint, in this constraint hanlder we only will check all and-constraints
    */
   SCIP_CALL( createAndAddLinearCons(scip, conshdlr, linvars, nlinvars, linvals, andress, nandconss, andcoefs, lhs, rhs, 
         initial, separate, enforce, FALSE/*check*/, propagate, local, modifiable, dynamic, removable, stickingatnode, 
         &lincons, &linconstype) );
   assert(lincons != NULL);
   assert(linconstype > SCIP_INVALIDCONS);

   /* create constraint data */
   /* checking for and-constraints will be FALSE, we check all information in this constraint handler */
   SCIP_CALL( consdataCreate(scip, conshdlr, &consdata, lincons, linconstype, andconss, andcoefs, nandconss,
         indvar, weight, issoftcons, intvar, lhs, rhs) );
   assert(consdata != NULL);
   
   /* free temporary memory */
   SCIPfreeBufferArray(scip, &andcoefs);
   SCIPfreeBufferArray(scip, &andress);
   SCIPfreeBufferArray(scip, &andconss);
   
   /* create constraint */
   SCIP_CALL( SCIPcreateCons(scip, cons, name, conshdlr, consdata, initial, separate, enforce, check, propagate,
         local, modifiable, dynamic, removable, stickingatnode) );
   
   return SCIP_OKAY;
}

/** @Note: you can only add a coefficient if the special type of linear constraint won't changed */
/** @todo: if adding a coefficient would change the type of the special linear constraint, we need to erase it and
 *         create a new linear constraint */
/** adds a variable to the pseudo boolean constraint (if it is not zero) */
SCIP_RETCODE SCIPaddCoefPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< constraint data */
   SCIP_VAR*const        var,                /**< variable of constraint entry */
   SCIP_Real const       val                 /**< coefficient of constraint entry */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(var != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      return SCIP_INVALIDDATA;
   }

   if( SCIPisZero(scip, val) )
      return SCIP_OKAY;
   
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   switch( consdata->linconstype )
   {
   case SCIP_LINEAR:
      SCIP_CALL( SCIPaddCoefLinear(scip, consdata->lincons, var, val) );
      break;
   case SCIP_LOGICOR:
      if( !SCIPisEQ(scip, val, 1.0) )
         return SCIP_INVALIDDATA;
      
      SCIP_CALL( SCIPaddCoefLogicor(scip, consdata->lincons, var) );
      break;
   case SCIP_KNAPSACK:
      if( !SCIPisIntegral(scip, val) || !SCIPisPositive(scip, val) )
         return SCIP_INVALIDDATA;
      
      SCIP_CALL( SCIPaddCoefKnapsack(scip, consdata->lincons, var, (SCIP_Longint) val) );
      break;
   case SCIP_SETPPC:
      if( !SCIPisEQ(scip, val, 1.0) )
         return SCIP_INVALIDDATA;
      
      SCIP_CALL( SCIPaddCoefSetppc(scip, consdata->lincons, var) );
      break;
#if 0
   case SCIP_EQKNAPSACK:
      if( !SCIPisIntegral(scip, val) || !SCIPisPositive(scip, val) )
         return SCIP_INVALIDDATA;
      
      SCIP_CALL( SCIPaddCoefEQKnapsack(scip, consdata->lincons, var, (SCIP_Longint) val) );
      break;
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   consdata->propagated = FALSE;
   consdata->presolved = FALSE;
   consdata->cliquesadded = FALSE;

   return SCIP_OKAY;
}


/** @Note: you can only add a coefficient if the special type of linear constraint won't changed */
/** @todo: if adding a coefficient would change the type of the special linear constraint, we need to erase it and
 *         create a new linear constraint */
/** adds nonlinear term to pseudo boolean constraint (if it is not zero) */
extern
SCIP_RETCODE SCIPaddTermPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_VAR**const       vars,               /**< variables of the nonlinear term */
   int const             nvars,              /**< number of variables of the nonlinear term */
   SCIP_Real const       val                 /**< coefficient of constraint entry */
   )
{
   assert(scip != NULL);
   assert(cons != NULL);
   assert(nvars == 0 || vars != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( addCoefTerm(scip, cons, vars, nvars, val) );

   return SCIP_OKAY;
}

/** gets indicator variable of pseudoboolean constraint, or NULL if there is no */
SCIP_VAR* SCIPgetIndVarPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      SCIPABORT();
   }

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->indvar;
}

/** gets linear constraint of pseudoboolean constraint */
SCIP_CONS* SCIPgetLinearConsPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      SCIPABORT();
   }

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->lincons;
}

/** gets type of linear constraint of pseudoboolean constraint */
SCIP_LINEARCONSTYPE SCIPgetLinearConsTypePseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      SCIPABORT();
   }

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->linconstype;
}

/** gets number of linear variables without artificial terms variables of pseudoboolean constraint */
int SCIPgetNLinVarsWithoutAndPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< pseudoboolean constraint */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      SCIPABORT();
   }

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->nlinvars;
}

/** gets linear constraint of pseudoboolean constraint */
SCIP_RETCODE SCIPgetLinDatasWithoutAndPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_VAR**const       linvars,            /**< array to store and-constraints */
   SCIP_Real*const       lincoefs,           /**< array to store and-coefficients */
   int*const             nlinvars            /**< pointer to store the required array size for and-constraints, have to
                                              *   be initialized with size of given array */ 
   )
{
   SCIP_CONSDATA* consdata;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   int nvars;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(nlinvars != NULL);
   assert(*nlinvars == 0 || linvars != NULL);
   assert(*nlinvars == 0 || lincoefs != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      SCIPABORT();
   }

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   if( *nlinvars < consdata->nlinvars )
   {
      *nlinvars = consdata->nlinvars;
      return SCIP_OKAY;
   }

   /* gets number of variables in linear constraint */
   SCIP_CALL( getLinearConsNVars(scip, consdata->lincons, consdata->linconstype, &nvars) );

   /* allocate temporary memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nvars) );
      
   /* get variables and coefficient of linear constraint */
   SCIP_CALL( getLinearConsVarsData(scip, consdata->lincons, consdata->linconstype, vars, coefs, &nvars) );
      
   /* calculate all not artificial linear variables */
   SCIP_CALL( getLinVarsAndAndRess(scip, cons, vars, coefs, nvars, linvars, lincoefs, nlinvars, NULL, NULL, NULL) );

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}


/** gets and-constraints of pseudoboolean constraint */
SCIP_RETCODE SCIPgetAndDatasPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< pseudoboolean constraint */
   SCIP_CONS**const      andconss,           /**< array to store and-constraints */
   SCIP_Real*const       andcoefs,           /**< array to store and-coefficients */
   int*const             nandconss           /**< pointer to store the required array size for and-constraints, have to
                                              *   be initialized with size of given array */ 
   )
{
   SCIP_CONSDATA* consdata;
   int c;
   
   assert(scip != NULL);
   assert(cons != NULL);
   assert(nandconss != NULL);
   assert(*nandconss == 0 || andconss != NULL);
   assert(*nandconss == 0 || andcoefs != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      SCIPABORT();
   }

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   if( *nandconss < consdata->nconsanddatas )
   {
      *nandconss = consdata->nconsanddatas;
      return SCIP_OKAY;
   }

   *nandconss = consdata->nconsanddatas;
   assert(*nandconss == 0 || consdata->consanddatas != NULL);

   for( c = *nandconss - 1; c >= 0; --c )
   {
      assert(consdata->consanddatas[c] != NULL);
      andconss[c] = consdata->consanddatas[c]->cons;
      andcoefs[c] = consdata->andcoefs[c];
      assert(andconss[c] != NULL);
   }

   return SCIP_OKAY;
}

/** gets number of and constraints of pseudoboolean constraint */
int SCIPgetNAndsPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< constraint data */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      SCIPABORT();
   }

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->nconsanddatas;
}

/** @Note: you can only changed the left hand side if the special type of linear constraint won't changed */
/** @todo: if changing the left hand side would change the type of the special linear constraint, we need to erase it
 *         and create a new linear constraint */
/** changes left hand side of pseudoboolean constraint */
SCIP_RETCODE SCIPchgLhsPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< constraint data */
   SCIP_Real const       lhs                 /**< new left hand side */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      return SCIP_INVALIDDATA;
   }

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   switch( consdata->linconstype )
   {
   case SCIP_LINEAR:
      SCIP_CALL( chgLhs(scip, cons, lhs) );
   case SCIP_LOGICOR:
   case SCIP_KNAPSACK:
   case SCIP_SETPPC:
      SCIPerrorMessage("changing left hand side only allowed on standard linear constraint \n");
      return SCIP_INVALIDDATA;
#if 0
   case SCIP_EQKNAPSACK:
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** @Note: you can only changed the right hand side if the special type of linear constraint won't changed */
/** @todo: if changing the right hand side would change the type of the special linear constraint, we need to erase it
 *         and create a new linear constraint */
/** changes right hand side of pseudoboolean constraint */
SCIP_RETCODE SCIPchgRhsPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons,               /**< constraint data */
   SCIP_Real const       rhs                 /**< new right hand side */
   )
{
   SCIP_CONSDATA* consdata;

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      return SCIP_INVALIDDATA;
   }

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   switch( consdata->linconstype )
   {
   case SCIP_LINEAR:
      SCIP_CALL( chgRhs(scip, cons, rhs) );
   case SCIP_LOGICOR:
   case SCIP_KNAPSACK:
   case SCIP_SETPPC:
      SCIPerrorMessage("changing right hand side only allowed on standard linear constraint \n");
      return SCIP_INVALIDDATA;
#if 0
   case SCIP_EQKNAPSACK:
#endif
   default:
      SCIPerrorMessage("unknown linear constraint type\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** get left hand side of pseudoboolean constraint */
SCIP_Real SCIPgetLhsPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< pseudoboolean constraint */
   )
{
   SCIP_CONSDATA* consdata;

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      return SCIP_INVALIDDATA;
   }

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   
   return consdata->lhs;
}

/** get right hand side of pseudoboolean constraint */
SCIP_Real SCIPgetRhsPseudoboolean(
   SCIP*const            scip,               /**< SCIP data structure */
   SCIP_CONS*const       cons                /**< pseudoboolean constraint */
   )
{
   SCIP_CONSDATA* consdata;

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), CONSHDLR_NAME) != 0 )
   {
      SCIPerrorMessage("constraint is not pseudo boolean\n");
      return SCIP_INVALIDDATA;
   }

#ifdef SCIP_DEBUG
   SCIP_CALL( checkConsConsistency(scip, cons) );
#endif

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   
   return consdata->rhs;
}
