/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*  Copyright (c) 2002-2023 Zuse Institute Berlin (ZIB)                      */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SCIP; see the file LICENSE. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   symmetry_orbital.c
 * @ingroup OTHER_CFILES
 * @brief  methods for handling symmetries by orbital reduction
 * @author Jasper van Doornmalen
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "blockmemshell/memory.h"
#include "scip/symmetry_orbital.h"
#include "scip/pub_cons.h"
#include "scip/pub_message.h"
#include "scip/pub_var.h"
#include "scip/struct_var.h"
#include "scip/type_var.h"
#include "scip/scip.h"
#include "scip/scip_branch.h"
#include "scip/scip_conflict.h"
#include "scip/scip_cons.h"
#include "scip/scip_copy.h"
#include "scip/scip_cut.h"
#include "scip/scip_general.h"
#include "scip/scip_lp.h"
#include "scip/scip_mem.h"
#include "scip/scip_message.h"
#include "scip/scip_numerics.h"
#include "scip/scip_param.h"
#include "scip/scip_prob.h"
#include "scip/scip_probing.h"
#include "scip/scip_sol.h"
#include "scip/scip_var.h"
#include "scip/debug.h"
#include "scip/struct_scip.h"
#include "scip/struct_mem.h"
#include "scip/struct_tree.h"
#include "scip/symmetry.h"
#include "scip/event_shadowtree.h"
#include <ctype.h>
#include <string.h>
#include <memory.h>


/* event handler properties */
#define EVENTHDLR_SYMMETRY_NAME    "symmetry_orbital"
#define EVENTHDLR_SYMMETRY_DESC    "filter global variable bound reduction event handler for orbital reduction"


/*
 * Data structures
 */


/** data for orbital reduction component propagator */
struct OrbitalReductionComponentData
{
   SCIP_NODE*            lastnode;           /**< last node processed by orbital reduction component */
   SCIP_Real*            globalvarlbs;       /**< global variable lower bounds until before branching starts */
   SCIP_Real*            globalvarubs;       /**< global variable upper bounds until before branching starts */
   int**                 perms;              /**< the permutations for orbital reduction */
   int                   nperms;             /**< the number of permutations in perms */
   SCIP_VAR**            permvars;           /**< array consisting of the variables of this component */
   int                   npermvars;          /**< number of vars in this component */
   SCIP_HASHMAP*         permvarmap;         /**< map of variables to indices in permvars array */

   SCIP_Bool             symmetrybrokencomputed; /**< whether the symmetry broken information is computed already */
   int*                  symbrokenvarids;    /**< variables to be stabilized because the symmetry is globally broken */
   int                   nsymbrokenvarids;   /**< symbrokenvarids array length, is 0 iff symbrokenvarids is NULL */
};
typedef struct OrbitalReductionComponentData ORCDATA;


/** data for orbital reduction propagator */
struct SCIP_OrbitalReductionData
{
   SCIP_EVENTHDLR*       shadowtreeeventhdlr;/**< eventhandler for the shadow tree data structure */
   SCIP_EVENTHDLR*       globalfixeventhdlr; /**< event handler for handling global variable bound reductions */

   ORCDATA**             componentdatas;     /**< array of pointers to individual components for orbital reduction */
   int                   ncomponents;        /**< number of orbital reduction datas in array */
   int                   maxncomponents;     /**< allocated orbital reduction datas array size */
   int                   nred;               /**< total number of reductions */
};


/*
 * Local methods
 */


/** identifies the orbits at which symmetry is broken according to the global bounds
 *
 *  An example of a symmetry-breaking constraint is cons_components.
 */
static
SCIP_RETCODE identifyOrbitalSymmetriesBroken(
   SCIP*                 scip,               /**< pointer to SCIP data structure */
   ORCDATA*              orcdata             /**< pointer to data for orbital reduction data */
)
{
   SCIP_DISJOINTSET* orbitset;
   int i;
   int j;
   int p;
   int* perm;
   int* varorbitids;
   int* varorbitidssort;
   int orbitbegin;
   int orbitend;
   int orbitid;
   int maxnsymbrokenvarids;
   SCIP_Real orbitglb;
   SCIP_Real orbitgub;
   SCIP_Bool orbitsymbroken;

   assert( scip != NULL );
   assert( orcdata != NULL );
   assert( !orcdata->symmetrybrokencomputed );
   orcdata->symbrokenvarids = NULL;
   orcdata->nsymbrokenvarids = 0;
   maxnsymbrokenvarids = 0;

   /* determine all orbits */
   SCIP_CALL( SCIPcreateDisjointset(scip, &orbitset, orcdata->npermvars) );
   for (p = 0; p < orcdata->nperms; ++p)
   {
      perm = orcdata->perms[p];
      assert( perm != NULL );

      for (i = 0; i < orcdata->npermvars; ++i)
      {
         j = perm[i];
         if ( i != j )
            SCIPdisjointsetUnion(orbitset, i, j, FALSE);
      }
   }

#ifndef NDEBUG
   for (i = 0; i < orcdata->npermvars; ++i)
   {
      assert( SCIPvarGetLbGlobal(orcdata->permvars[i]) == orcdata->globalvarlbs[i] );
      assert( SCIPvarGetUbGlobal(orcdata->permvars[i]) == orcdata->globalvarubs[i] );
   }
#endif

   /* sort all orbits */
   SCIP_CALL( SCIPallocBufferArray(scip, &varorbitids, orcdata->npermvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varorbitidssort, orcdata->npermvars) );
   for (i = 0; i < orcdata->npermvars; ++i)
      varorbitids[i] = SCIPdisjointsetFind(orbitset, i);
   SCIPsort(varorbitidssort, SCIPsortArgsortInt, varorbitids, orcdata->npermvars);

   /* iterate over all orbits and get the maximal orbit lower bound and minimal orbit upper bound */
   for (orbitbegin = 0; orbitbegin < orcdata->npermvars; orbitbegin = orbitend)
   {
      /* get id of the orbit */
      orbitid = varorbitids[varorbitidssort[orbitbegin]];

      /* the orbit must have the same bounds */
      orbitsymbroken = FALSE;
      j = varorbitids[orbitbegin];
      orbitglb = orcdata->globalvarlbs[j];
      orbitgub = orcdata->globalvarubs[j];
      for (i = orbitbegin + 1; i < orcdata->npermvars; ++i)
      {
         j = varorbitidssort[i];

         /* stop if j is not the element in the orbit, then it is part of the next orbit */
         if ( varorbitids[j] != orbitid )
            break;

         if ( !orbitsymbroken )
         {
            if ( !EQ(scip, orbitglb, orcdata->globalvarlbs[j]) || !EQ(scip, orbitgub, orcdata->globalvarubs[j]) )
            {
               orbitsymbroken = TRUE;
               break;
            }
         }
      }
      /* the loop above has terminated, so i is either orcdata->npermvars or varorbitidssort[i] is in the next orbit,
       * and orbitglb and orbitgub are the maximal global lower bound and minimal global upper bound in orbit orbitid */
      orbitend = i;

      /* symmetry is broken within this orbit if the intersection of the global variable domains are empty */
      if ( orbitsymbroken )
      {
         /* add all variable ids in the orbit to the symbrokenvarids array: resize if needed */
         if ( orcdata->nsymbrokenvarids + orbitend - orbitbegin > maxnsymbrokenvarids )
         {
            int newsize;

            newsize = SCIPcalcMemGrowSize(scip, orcdata->nsymbrokenvarids + orbitend - orbitbegin);
            assert( newsize >= 0 );

            if ( orcdata->nsymbrokenvarids == 0 )
            {
               assert( orcdata->symbrokenvarids == NULL );
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &orcdata->symbrokenvarids, newsize) );
            }
            else
            {
               assert( orcdata->symbrokenvarids != NULL );
               SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &orcdata->symbrokenvarids,
                  maxnsymbrokenvarids, newsize) );
            }

            maxnsymbrokenvarids = newsize;
         }

         /* add all variable ids in the orbit to the symbrokenvarids array: add */
         for (i = orbitbegin; i < orbitend; ++i)
         {
            j = varorbitidssort[i];
            assert( varorbitids[j] == orbitid );
            assert( orcdata->nsymbrokenvarids < maxnsymbrokenvarids );
            orcdata->symbrokenvarids[orcdata->nsymbrokenvarids++] = j;
         }
      }
   }

   /* shrink the allocated array size to the actually needed size */
   assert( orcdata->nsymbrokenvarids <= maxnsymbrokenvarids );
   if ( orcdata->nsymbrokenvarids > 0 && orcdata->nsymbrokenvarids < maxnsymbrokenvarids )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &orcdata->symbrokenvarids,
         maxnsymbrokenvarids, orcdata->nsymbrokenvarids) );
   }
   assert( (orcdata->nsymbrokenvarids == 0) == (orcdata->symbrokenvarids == NULL) );

   /* mark that this method is executed for the component */
   orcdata->symmetrybrokencomputed = TRUE;

   /* output information */
   if ( orcdata->nsymbrokenvarids > 0 )
   {
      SCIPwarningMessage(scip,
         "Orbital fixing symmetry for %p broken before symmetry. Requires fixing %d/%d affected variables.\n",
         (void*) orcdata, orcdata->nsymbrokenvarids, orcdata->npermvars);
   }

   SCIPfreeBufferArray(scip, &varorbitidssort);
   SCIPfreeBufferArray(scip, &varorbitids);
   SCIPfreeDisjointset(scip, &orbitset);

   return SCIP_OKAY;
}


/** populates chosenperms with a generating set of the symmetry group stabilizing the branching decisions
 *
 *  The symmetry subgroup considered is generated by all permutations where for all branching variables \f$x$
 *  with permuted variable \f$y$ for all possible variable assignments we have \f$x \leq y$.
 *  We restrict ourselves to testing this only for the group generators.
 */
static
SCIP_RETCODE orbitalReductionGetSymmetryStabilizerSubgroup(
   SCIP*                 scip,               /**< pointer to SCIP data structure */
   ORCDATA*              orcdata,            /**< pointer to data for orbital reduction data */
   int**                 chosenperms,        /**< pointer to permutations that are chosen */
   int*                  nchosenperms,       /**< pointer to store the number of chosen permutations */
   SCIP_Real*            varlbs,             /**< array of orcdata->permvars variable LBs. If NULL, use local bounds */
   SCIP_Real*            varubs,             /**< array of orcdata->permvars variable UBs. If NULL, use local bounds */
   int*                  branchedvarindices, /**< array of given branching decisions, in branching order */
   SCIP_Bool*            inbranchedvarindices, /**< array stating whether variable with index in orcdata->permvars is
                                                *   contained in the branching decisions. */
   int                   nbranchedvarindices /**< number of branching decisions */
)
{
   int i;
   int p;
   int* perm;
   int varid;
   int varidimage;

   assert( scip != NULL );
   assert( orcdata != NULL );
   assert( chosenperms != NULL );
   assert( nchosenperms != NULL );
   assert( (varlbs == NULL) == (varubs == NULL) );
   assert( branchedvarindices != NULL );
   assert( inbranchedvarindices != NULL );
   assert( nbranchedvarindices >= 0 );
   assert( orcdata->symmetrybrokencomputed );
   assert( (orcdata->nsymbrokenvarids == 0) == (orcdata->symbrokenvarids == NULL) );

   *nchosenperms = 0;

   for (p = 0; p < orcdata->nperms; ++p)
   {
      perm = orcdata->perms[p];

      /* make sure that the symmetry broken orbit variable indices are met with equality */
      for (i = 0; i < orcdata->nsymbrokenvarids; ++i)
      {
         varid = orcdata->symbrokenvarids[i];
         assert( varid >= 0 );
         assert( varid < orcdata->npermvars );
         assert( orcdata->permvars[varid] != NULL );
         varidimage = perm[varid];
         assert( varidimage >= 0 );
         assert( varidimage < orcdata->npermvars );
         assert( orcdata->permvars[varidimage] != NULL );

         /* branching variable is not affected by this permutation */
         if ( varidimage == varid )
            continue;

         /* the variables on which symmetry is broken must be permuted to entries with the same fixed value
          *
          * Because we check a whole orbit of the group and perm is part of it, it suffices to compare the upper bound
          * of varid with the lower bound of varidimage. Namely, for all indices i, \f$lb_i \leq ub_i\f$, so we get
          * a series of equalities yielding that all expressions must be the same:
          * \f$ub_i = lb_j <= ub_j = lb_{\cdots} <= \cdots = lb_j < ub_j \f$
          */
         if ( ! EQ(scip,
            varubs ? varubs[varid] : SCIPvarGetUbLocal(orcdata->permvars[varid]),
            varlbs ? varlbs[varidimage] : SCIPvarGetLbLocal(orcdata->permvars[varidimage]) )
         )
            break;
      }
      /* if the above loop is broken, this permutation does not qualify for the stabilizer */
      if ( i < orcdata->nsymbrokenvarids )
         continue;

      /* iterate over each branched variable and check */
      for (i = 0; i < nbranchedvarindices; ++i)
      {
         varid = branchedvarindices[i];
         assert( varid >= 0 );
         assert( varid < orcdata->npermvars );
         assert( orcdata->permvars[varid] != NULL );
         varidimage = perm[varid];
         assert( varidimage >= 0 );
         assert( varidimage < orcdata->npermvars );
         assert( orcdata->permvars[varidimage] != NULL );

         /* branching variable is not affected by this permutation */
         if ( varidimage == varid )
            continue;

         if ( GT(scip,
            varubs ? varubs[varid] : SCIPvarGetUbLocal(orcdata->permvars[varid]),
            varlbs ? varlbs[varidimage] : SCIPvarGetLbLocal(orcdata->permvars[varidimage]) )
         )
            break;
      }
      /* if the above loop is broken, this permutation does not qualify for the stabilizer */
      if ( i < nbranchedvarindices )
         continue;

      /* permutation qualifies for the stabilizer. Add permutation */
      chosenperms[(*nchosenperms)++] = perm;
   }

   return SCIP_OKAY;
}

/** using bisection, finds the minimal index k (frameleft <= k < frameright) such that ids[idssort[k]] >= findid
 *
 *  If for all k (frameleft <= k < frameright) holds ids[idssort[k]] < findid, returns frameright.
 */
static
int bisectSortedArrayFindFirstGEQ(
   int*               ids,                /**< int array with entries */
   int*               idssort,            /**< array of indices of ids that sort ids */
   int                frameleft,          /**< search in idssort for index range [frameleft, frameright) */
   int                frameright,         /**< search in idssort for index range [frameleft, frameright) */
   int                findid              /**< entry value to find */
)
{
   int center;
   int id;

#ifndef NDEBUG
   int origframeleft;
   int origframeright;
   origframeleft = frameleft;
   origframeright = frameright;
#endif

   assert( ids != NULL );
   assert( idssort != NULL );
   assert( frameleft >= 0 );
   assert( frameright >= frameleft );

   /* empty frame case */
   if ( frameright == frameleft )
      return frameright;

   while (frameright - frameleft >= 2)
   {
      /* split [frameleft, frameright) in [frameleft, center) and [center, frameright) */
      center = frameleft + ((frameright - frameleft) / 2);
      assert( center > frameleft );
      assert( center < frameright );
      id = idssort[center];
      if ( ids[id] < findid )
      {
         /* first instance greater or equal to findid is in [center, frameright) */
         frameleft = center;
      }
      else
      {
         /* first instance greater or equal to findid is in [frameleft, center) */
         frameright = center;
      }
   }

   assert( frameright - frameleft == 1 );
   id = idssort[frameleft];
   if ( ids[id] < findid )
      ++frameleft;

   assert( frameleft >= origframeleft );
   assert( frameright <= origframeright );
   assert( frameleft >= origframeright || ids[idssort[frameleft]] >= findid );
   assert( frameleft - 1 < origframeleft || ids[idssort[frameleft - 1]] < findid );
   return frameleft;
}


/** applies the orbital reduction steps for precomputed orbits
 *
 *  Either use the local variable bounds, or variable bounds determined by the @param varlbs and @param varubs arrays.
 *  @pre @param varubs is NULL if and only if @param varlbs is NULL.
 */
static
SCIP_RETCODE applyOrbitalReductionPart(
   SCIP*                 scip,               /**< pointer to SCIP data structure */
   ORCDATA*              orcdata,            /**< pointer to data for orbital reduction data */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is detected */
   int*                  nred,               /**< pointer to store the number of determined domain reductions */
   int*                  varorbitids,        /**< array specifying the orbit IDs for variables in array orcdata->vars */
   int*                  varorbitidssort,    /**< an index array that sorts the varorbitids array */
   SCIP_Real*            varlbs,             /**< array of lower bounds for variable array orcdata->vars to compute with
                                              *   or NULL, if local bounds are used */
   SCIP_Real*            varubs              /**< array of upper bounds for variable array orcdata->vars to compute with
                                              *   or NULL, if local bounds are used. */
)
{
   int i;
   int varid;
   int orbitid;
   int orbitbegin;
   int orbitend;
   SCIP_Real orbitlb;
   SCIP_Real orbitub;
   SCIP_Real lb;
   SCIP_Real ub;

   assert( scip != NULL );
   assert( orcdata != NULL );
   assert( infeasible != NULL );
   assert( nred != NULL );
   assert( varorbitids != NULL );
   assert( varorbitidssort != NULL );
   assert( ( varlbs == NULL ) == ( varubs == NULL ) );

   /* infeasible and nred are defined by the function that calls this function,
    * and this function only gets called if no infeasibility is found so far.
    */
   assert( !*infeasible );
   assert( *nred >= 0 );

   for (orbitbegin = 0; orbitbegin < orcdata->npermvars; orbitbegin = orbitend)
   {
      /* get id of the orbit, and scan how large the orbit is */
      orbitid = varorbitids[varorbitidssort[orbitbegin]];
      for (orbitend = orbitbegin + 1; orbitend < orcdata->npermvars; ++orbitend)
      {
         if ( varorbitids[varorbitidssort[orbitend]] != orbitid )
            break;
      }

      /* orbits consisting of only one element cannot yield reductions */
      if ( orbitend - orbitbegin <= 1 )
         continue;

      /* get upper and lower bounds in orbit */
      orbitlb = -INFINITY;
      orbitub = INFINITY;
      for (i = orbitbegin; i < orbitend; ++i)
      {
         varid = varorbitidssort[i];
         assert( varid >= 0 );
         assert( varid < orcdata->npermvars );
         assert( orcdata->permvars[varid] != NULL );

         lb = varlbs ? varlbs[varid] : SCIPvarGetLbLocal(orcdata->permvars[varid]);
         if ( GT(scip, lb, orbitlb) )
            orbitlb = lb;
         ub = varubs ? varubs[varid] : SCIPvarGetUbLocal(orcdata->permvars[varid]);
         if ( LT(scip, ub, orbitub) )
            orbitub = ub;
      }

      /* if bounds are incompatible, infeasibility is detected */
      if ( GT(scip, orbitlb, orbitub) )
      {
         *infeasible = TRUE;
         return SCIP_OKAY;
      }
      assert( LE(scip, orbitlb, orbitub) );

      /* update variable bounds to be in this range */
      for (i = orbitbegin; i < orbitend; ++i)
      {
         varid = varorbitidssort[i];
         assert( varid >= 0 );
         assert( varid < orcdata->npermvars );

         if ( varlbs != NULL )
         {
            assert( LE(scip, varlbs[varid], orbitlb) );
            varlbs[varid] = orbitlb;
         }
         if ( !SCIPisInfinity(scip, -orbitlb) &&
            LT(scip, SCIPvarGetLbLocal(orcdata->permvars[varid]), orbitlb) )
         {
            SCIP_Bool tightened;
            SCIP_CALL( SCIPtightenVarLb(scip, orcdata->permvars[varid], orbitlb, TRUE, infeasible, &tightened) );

            /* propagator detected infeasibility in this node */
            if ( *infeasible )
               return SCIP_OKAY;
            assert( tightened );
            *nred += 1;
         }

         if ( varubs != NULL )
         {
            assert( GE(scip, varubs[varid], orbitub) );
            varubs[varid] = orbitub;
         }
         if ( !SCIPisInfinity(scip, orbitub) &&
            GT(scip, SCIPvarGetUbLocal(orcdata->permvars[varid]), orbitub) )
         {
            SCIP_Bool tightened;
            SCIP_CALL( SCIPtightenVarUb(scip, orcdata->permvars[varid], orbitub, TRUE, infeasible, &tightened) );

            /* propagator detected infeasibility in this node */
            if ( *infeasible )
               return SCIP_OKAY;
            assert( tightened );
            *nred += 1;
         }
      }
   }
   assert( !*infeasible );
   return SCIP_OKAY;
}


/** orbital reduction, the orbital branching part */
static
SCIP_RETCODE applyOrbitalBranchingPropagations(
   SCIP*                 scip,               /**< pointer to SCIP data structure */
   ORCDATA*              orcdata,            /**< pointer to data for orbital reduction data */
   SCIP_SHADOWTREE*      shadowtree,         /**< pointer to shadow tree */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is detected */
   int*                  nred                /**< pointer to store the number of determined domain reductions */
)
{
   SCIP_NODE* focusnode;
   SCIP_NODE* parentnode;
   SCIP_SHADOWNODE* shadowfocusnode;
   SCIP_SHADOWNODE* tmpshadownode;
   SCIP_SHADOWNODE** rootedshadowpath;
   int pathlength;
   int depth;
   int branchstep;
   int i;
   SCIP_Real* varlbs;
   SCIP_Real* varubs;
   SCIP_SHADOWBOUNDUPDATE* update;
   int* branchedvarindices;
   SCIP_Bool* inbranchedvarindices;
   int nbranchedvarindices;
   int varid;
   SCIP_SHADOWBOUNDUPDATE* branchingdecision;
   int branchingdecisionvarid;
   int** chosenperms;
   int* perm;
   int nchosenperms;
   int p;
   int* varorbitids;
   int* varorbitidssort;
   int idx;
   int orbitbegin;
   int orbitend;
   SCIP_DISJOINTSET* orbitset;
   int orbitsetcomponentid;

   assert( scip != NULL );
   assert( orcdata != NULL );
   assert( shadowtree != NULL );
   assert( infeasible != NULL );
   assert( nred != NULL );

   /* infeasible and nred are defined by the function that calls this function,
    * and this function only gets called if no infeasibility is found so far.
    */
   assert( !*infeasible );
   assert( *nred >= 0 );

   focusnode = SCIPgetFocusNode(scip);
   assert( focusnode == SCIPgetCurrentNode(scip) );
   assert( focusnode != NULL );

   /* do nothing if this method has already been called for this node */
   if ( orcdata->lastnode == focusnode )
      return SCIP_OKAY;

   orcdata->lastnode = focusnode;
   parentnode = SCIPnodeGetParent(focusnode);

   /* the root node has not been generated by branching decisions */
   if ( parentnode == NULL )
      return SCIP_OKAY;

   shadowfocusnode = SCIPshadowTreeGetShadowNode(shadowtree, focusnode);
   assert( shadowfocusnode != NULL );

   /* get the rooted path */
   /* @todo add depth field to shadow tree node to improve efficiency */
   pathlength = 0;
   tmpshadownode = shadowfocusnode;
   do
   {
      tmpshadownode = tmpshadownode->parent;
      ++pathlength;
   }
   while ( tmpshadownode != NULL );

   SCIP_CALL( SCIPallocBufferArray(scip, &rootedshadowpath, pathlength) );
   i = pathlength;
   tmpshadownode = shadowfocusnode;
   while ( i > 0 )
   {
      rootedshadowpath[--i] = tmpshadownode;
      assert( tmpshadownode != NULL );
      tmpshadownode = tmpshadownode->parent;
   }
   assert( tmpshadownode == NULL );
   assert( i == 0 );

   /* replay bound reductions and propagations made until just before the focusnode */
   assert( orcdata->npermvars > 0 );  /* if it's 0, then we do not have to do anything at all */

   SCIP_CALL( SCIPallocBufferArray(scip, &varlbs, orcdata->npermvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varubs, orcdata->npermvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &branchedvarindices, orcdata->npermvars) );
   SCIP_CALL( SCIPallocCleanBufferArray(scip, &inbranchedvarindices, orcdata->npermvars) );

   /* start with the bounds found after computing the symmetry group */
   for (i = 0; i < orcdata->npermvars; ++i)
      varlbs[i] = orcdata->globalvarlbs[i];
   for (i = 0; i < orcdata->npermvars; ++i)
      varubs[i] = orcdata->globalvarubs[i];

   nbranchedvarindices = 0;
   for (depth = 0; depth < pathlength - 1; ++depth)
   {
      tmpshadownode = rootedshadowpath[depth];

      /* receive propagations */
      for (i = 0; i < tmpshadownode->npropagations; ++i)
      {
         update = &(tmpshadownode->propagations[i]);
         varid = SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) update->var);
         assert( varid < orcdata->npermvars || varid == INT_MAX );
         assert( varid >= 0 );
         if ( varid < orcdata->npermvars )
         {
            assert( LE(scip, varlbs[varid], varubs[varid]) );
            switch (update->boundchgtype)
            {
               case SCIP_BOUNDTYPE_LOWER:
                  assert( GE(scip, update->newbound, varlbs[varid]) );
                  varlbs[varid] = update->newbound;
                  break;
               case SCIP_BOUNDTYPE_UPPER:
                  assert( LE(scip, update->newbound, varubs[varid]) );
                  varubs[varid] = update->newbound;
                  break;
               default:
                  assert( FALSE );
            }
            assert( LE(scip, varlbs[varid], varubs[varid]) );
         }
      }

      /* receive variable indices of branched variables */
      for (i = 0; i < tmpshadownode->nbranchingdecisions; ++i)
      {
         update = &(tmpshadownode->branchingdecisions[i]);
         varid = SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) update->var);
         assert( varid < orcdata->npermvars || varid == INT_MAX );
         assert( varid >= 0 );
         if ( varid < orcdata->npermvars )
         {
            if ( inbranchedvarindices[varid] )
               continue;
            branchedvarindices[nbranchedvarindices++] = varid;
            inbranchedvarindices[varid] = TRUE;
         }
      }
   }

   /* determine symmetry group at this point, apply branched variable, apply orbital branching for this
    *
    * The branching variables are applied one-after-the-other.
    * So, the group before branching is determined, orbital branching to the branching variable, then the branching
    * variable is applied, and possibly repeated for other branching variables.
    */
   SCIP_CALL( SCIPallocBufferArray(scip, &chosenperms, orcdata->nperms) );
   for (branchstep = 0; branchstep < shadowfocusnode->nbranchingdecisions; ++branchstep)
   {
      branchingdecision = &(shadowfocusnode->branchingdecisions[branchstep]);
      branchingdecisionvarid = SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) branchingdecision->var);
      assert( branchingdecisionvarid < orcdata->npermvars || branchingdecisionvarid == INT_MAX );
      assert( branchingdecisionvarid >= 0 );

      /* branching decision will not have an effect on this */
      if ( branchingdecisionvarid >= orcdata->npermvars )
         continue;
      assert( branchingdecisionvarid >= 0 && branchingdecisionvarid < orcdata->npermvars );
      assert( branchingdecision->boundchgtype == SCIP_BOUNDTYPE_LOWER ?
         LE(scip, varlbs[branchingdecisionvarid], branchingdecision->newbound) :
         GE(scip, varubs[branchingdecisionvarid], branchingdecision->newbound) );
      assert( LE(scip, varlbs[branchingdecisionvarid], varubs[branchingdecisionvarid]) );

      /* get the generating set of permutations of a subgroup of a stabilizing symmetry subgroup.
       *
       * Note: All information about branching decisions is kept in varlbs, varubs, and the branchedvarindices.
       */
      SCIP_CALL( orbitalReductionGetSymmetryStabilizerSubgroup(scip, orcdata, chosenperms, &nchosenperms,
         varlbs, varubs, branchedvarindices, inbranchedvarindices, nbranchedvarindices) );

      /* compute orbit containing branching var */
      SCIP_CALL( SCIPcreateDisjointset(scip, &orbitset, orcdata->npermvars) );

      /* put elements mapping to each other in same orbit */
      /* @todo a potential performance hazard; quadratic time */
      for (p = 0; p < nchosenperms; ++p)
      {
         perm = chosenperms[p];
         for (i = 0; i < orcdata->npermvars; ++i)
         {
            if ( i != perm[i] )
               SCIPdisjointsetUnion(orbitset, i, perm[i], FALSE);
         }
      }

      /* 1. ensure that the bounds are tightest possible just before the branching step (orbital reduction step)
       *
       * If complete propagation was applied in the previous node,
       * then all variables in the same orbit have the same bounds just before branching,
       * so the bounds of the branching variable should be the tightest in its orbit by now.
       * It is possible that that is not the case. In that case, we do it here.
       */
      SCIP_CALL( SCIPallocBufferArray(scip, &varorbitids, orcdata->npermvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &varorbitidssort, orcdata->npermvars) );
      for (i = 0; i < orcdata->npermvars; ++i)
         varorbitids[i] = SCIPdisjointsetFind(orbitset, i);
      SCIPsort(varorbitidssort, SCIPsortArgsortInt, varorbitids, orcdata->npermvars);

      /* apply orbital reduction to these orbits */
      SCIP_CALL( applyOrbitalReductionPart(scip, orcdata, infeasible, nred, varorbitids,
         varorbitidssort, varlbs, varubs) );
      if ( *infeasible )
         goto FREE;
      assert( !*infeasible );

      /* 2. apply branching step to varlbs or varubs array
       *
       * Due to the steps above, it is possible that the branching step is redundant or infeasible.
       */
      assert( LE(scip, varlbs[branchingdecisionvarid], varubs[branchingdecisionvarid]) );
      switch (branchingdecision->boundchgtype)
      {
         case SCIP_BOUNDTYPE_LOWER:
            /* incompatible upper bound */
            if ( GT(scip, branchingdecision->newbound, varubs[branchingdecisionvarid]) )
            {
               *infeasible = TRUE;
               goto FREE;
            }

            assert( LE(scip, varlbs[branchingdecisionvarid], branchingdecision->newbound) );
            varlbs[branchingdecisionvarid] = branchingdecision->newbound;
            break;
         case SCIP_BOUNDTYPE_UPPER:
            /* incompatible lower bound */
            if ( LT(scip, branchingdecision->newbound, varlbs[branchingdecisionvarid]) )
            {
               *infeasible = TRUE;
               goto FREE;
            }

            assert( GE(scip, varubs[branchingdecisionvarid], branchingdecision->newbound) );
            varubs[branchingdecisionvarid] = branchingdecision->newbound;
            break;
         default:
            assert( FALSE );
      }

      /* 3. propagate that branching variable is >= the variables in its orbit
       *
       * Also apply the updates to the variable arrays
       */

      /* get the orbit of the branching variable */
      orbitsetcomponentid = SCIPdisjointsetFind(orbitset, branchingdecisionvarid);

      /* find the orbit in the sorted array of orbits. npermvars can be huge, so use bisection. */
      orbitbegin = bisectSortedArrayFindFirstGEQ(varorbitids, varorbitidssort, 0, orcdata->npermvars,
         orbitsetcomponentid);
      assert( orbitbegin >= 0 && orbitbegin < orcdata->npermvars );
      assert( varorbitids[varorbitidssort[orbitbegin]] == orbitsetcomponentid );
      assert( orbitbegin == 0 || varorbitids[varorbitidssort[orbitbegin - 1]] < orbitsetcomponentid );

      orbitend = bisectSortedArrayFindFirstGEQ(varorbitids, varorbitidssort, orbitbegin + 1, orcdata->npermvars,
         orbitsetcomponentid + 1);
      assert( orbitend > 0 && orbitend <= orcdata->npermvars && orbitend > orbitbegin );
      assert( orbitend == orcdata->npermvars || varorbitids[varorbitidssort[orbitend]] > orbitsetcomponentid );
      assert( varorbitids[varorbitidssort[orbitend - 1]] == orbitsetcomponentid );

      /* propagate that branching variable is >= the variables in its orbit */
      for (idx = orbitbegin; idx < orbitend; ++idx)
      {
         varid = varorbitidssort[idx];
         assert( varorbitids[varid] == orbitsetcomponentid );

         /* ignore current branching variable */
         if ( varid == branchingdecisionvarid )
            continue;

         /* is variable varid in the orbit? */
         if ( SCIPdisjointsetFind(orbitset, varid) != orbitsetcomponentid )
            continue;

         /* all variables in the same orbit have the same bounds just before branching,
          * due to orbital reduction. If that was not the case, these steps are applied just before applying
          * the branching step above. After the branching step, the branching variable bounds are most restricted.
          */
         assert( SCIPisInfinity(scip, -varlbs[branchingdecisionvarid])
            || GE(scip, varlbs[branchingdecisionvarid], varlbs[varid]) );
         assert( SCIPisInfinity(scip, varubs[branchingdecisionvarid])
            || LE(scip, varubs[branchingdecisionvarid], varubs[varid]) );
         /* bound changes already made could only have tightened the variable domains we are thinking about */
         assert( GE(scip, SCIPvarGetLbLocal(orcdata->permvars[varid]), varlbs[varid]) );
         assert( LE(scip, SCIPvarGetUbLocal(orcdata->permvars[varid]), varubs[varid]) );

         /* for branching variable x and variable y in its orbit, propagate x >= y. */
         /* modify UB of y-variables */
         assert( GE(scip, varubs[varid], varubs[branchingdecisionvarid]) );
         varubs[varid] = varubs[branchingdecisionvarid];
         if ( GT(scip, SCIPvarGetUbLocal(orcdata->permvars[varid]), varubs[branchingdecisionvarid]) )
         {
            SCIP_Bool tightened;
            SCIP_CALL( SCIPtightenVarUb(scip, orcdata->permvars[varid], varubs[branchingdecisionvarid], TRUE,
                  infeasible, &tightened) );

            /* propagator detected infeasibility in this node. */
            if ( *infeasible )
               goto FREE;
            assert( tightened );
            *nred += 1;
         }

         /* because variable domains are initially the same, the LB of the x-variables does not need to be modified. */
         assert( LE(scip, varlbs[varid], varlbs[branchingdecisionvarid]) );
      }

      FREE:
      SCIPfreeBufferArray(scip, &varorbitidssort);
      SCIPfreeBufferArray(scip, &varorbitids);
      SCIPfreeDisjointset(scip, &orbitset);

      if ( *infeasible )
         break;

      /* for the next branched variable at this node, if it's not already added,
       * mark the branching variable of this iteration as a branching variable. */
      if ( !inbranchedvarindices[branchingdecisionvarid] )
      {
         assert( nbranchedvarindices < orcdata->npermvars );
         branchedvarindices[nbranchedvarindices++] = branchingdecisionvarid;
         inbranchedvarindices[branchingdecisionvarid] = TRUE;
      }
   }
   SCIPfreeBufferArray(scip, &chosenperms);

   /* clean inbranchedvarindices array */
   for (i = 0; i < nbranchedvarindices; ++i)
   {
      varid = branchedvarindices[i];
      assert( varid >= 0 );
      assert( varid < orcdata->npermvars );
      assert( inbranchedvarindices[varid] );
      inbranchedvarindices[varid] = FALSE;
   }
#ifndef NDEBUG
   for (i = 0; i < orcdata->npermvars; ++i)
   {
      assert( inbranchedvarindices[i] == FALSE );
   }
#endif

   /* free everything */
   SCIPfreeCleanBufferArray(scip, &inbranchedvarindices);
   SCIPfreeBufferArray(scip, &branchedvarindices);
   SCIPfreeBufferArray(scip, &varubs);
   SCIPfreeBufferArray(scip, &varlbs);
   SCIPfreeBufferArray(scip, &rootedshadowpath);

   return SCIP_OKAY;
}

/** orbital reduction, the orbital reduction part */
static
SCIP_RETCODE applyOrbitalReductionPropagations(
   SCIP*                 scip,               /**< pointer to SCIP data structure */
   ORCDATA*              orcdata,            /**< pointer to data for orbital reduction data */
   SCIP_SHADOWTREE*      shadowtree,         /**< pointer to shadow tree */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is detected */
   int*                  nred                /**< pointer to store the number of determined domain reductions */
)
{
   SCIP_NODE* focusnode;
   SCIP_SHADOWNODE* shadowfocusnode;
   SCIP_SHADOWNODE* tmpshadownode;
   int i;
   SCIP_SHADOWBOUNDUPDATE* update;
   int* branchedvarindices;
   SCIP_Bool* inbranchedvarindices;
   int nbranchedvarindices;
   int varid;
   int** chosenperms;
   int* perm;
   int nchosenperms;
   int p;
   SCIP_DISJOINTSET* orbitset;
   int* varorbitids;
   int* varorbitidssort;

   assert( scip != NULL );
   assert( orcdata != NULL );
   assert( shadowtree != NULL );
   assert( infeasible != NULL );
   assert( nred != NULL );

   /* infeasible and nred are defined by the function that calls this function,
    * and this function only gets called if no infeasibility is found so far.
    */
   assert( !*infeasible );
   assert( *nred >= 0 );

   focusnode = SCIPgetFocusNode(scip);
   assert( focusnode == SCIPgetCurrentNode(scip) );
   assert( focusnode != NULL );

   shadowfocusnode = SCIPshadowTreeGetShadowNode(shadowtree, focusnode);
   assert( shadowfocusnode != NULL );

   /* get the branching variables until present, so including the branchings of the focusnode */
   assert( orcdata->npermvars > 0 );  /* if it's 0, then we do not have to do anything at all */

   SCIP_CALL( SCIPallocBufferArray(scip, &branchedvarindices, orcdata->npermvars) );
   SCIP_CALL( SCIPallocCleanBufferArray(scip, &inbranchedvarindices, orcdata->npermvars) );

   nbranchedvarindices = 0;
   tmpshadownode = shadowfocusnode;
   while ( tmpshadownode != NULL )
   {
      /* receive variable indices of branched variables */
      for (i = 0; i < tmpshadownode->nbranchingdecisions; ++i)
      {
         update = &(tmpshadownode->branchingdecisions[i]);
         varid = SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) update->var);
         assert( varid < orcdata->npermvars || varid == INT_MAX );
         assert( varid >= 0 );
         if ( varid < orcdata->npermvars )
         {
            if ( inbranchedvarindices[varid] )
               continue;
            branchedvarindices[nbranchedvarindices++] = varid;
            inbranchedvarindices[varid] = TRUE;
         }
      }
      tmpshadownode = tmpshadownode->parent;
   }

   /* 1. compute the orbit of the branching variable of the stabilized symmetry subgroup at this point. */
   /* 1.1. identify the permutations of the symmetry group that are permitted */
   SCIP_CALL( SCIPallocBufferArray(scip, &chosenperms, orcdata->nperms) );
   SCIP_CALL( orbitalReductionGetSymmetryStabilizerSubgroup(scip, orcdata, chosenperms, &nchosenperms,
      NULL, NULL, branchedvarindices, inbranchedvarindices, nbranchedvarindices) );
   assert( nchosenperms >= 0 );

   /* no reductions can be yielded by orbital reduction if the group is trivial */
   if ( nchosenperms == 0 )
      goto FREE;

   /* 1.2. compute orbits of this subgroup */
   SCIP_CALL( SCIPcreateDisjointset(scip, &orbitset, orcdata->npermvars) );

   /* put elements mapping to each other in same orbit */
   /* @todo this is O(nchosenperms * npermvars), which is a potential performance bottleneck.
      Alternative: precompute support per permutation at initialization, and iterate over these.*/
   for (p = 0; p < nchosenperms; ++p)
   {
      perm = chosenperms[p];
      for (i = 0; i < orcdata->npermvars; ++i)
      {
         if ( i != perm[i] )
            SCIPdisjointsetUnion(orbitset, i, perm[i], FALSE);
      }
   }

   /* 2. for each orbit, take the intersection of the domains */
   SCIP_CALL( SCIPallocBufferArray(scip, &varorbitids, orcdata->npermvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varorbitidssort, orcdata->npermvars) );
   for (i = 0; i < orcdata->npermvars; ++i)
      varorbitids[i] = SCIPdisjointsetFind(orbitset, i);
   SCIPsort(varorbitidssort, SCIPsortArgsortInt, varorbitids, orcdata->npermvars);

   /* apply orbital reduction to these orbits */
   SCIP_CALL( applyOrbitalReductionPart(scip, orcdata, infeasible, nred, varorbitids, varorbitidssort, NULL, NULL) );

   SCIPfreeBufferArray(scip, &varorbitidssort);
   SCIPfreeBufferArray(scip, &varorbitids);
   SCIPfreeDisjointset(scip, &orbitset);
FREE:
   SCIPfreeBufferArray(scip, &chosenperms);

   /* clean inbranchedvarindices array */
   for (i = 0; i < nbranchedvarindices; ++i)
   {
      varid = branchedvarindices[i];
      assert( varid >= 0 );
      assert( varid < orcdata->npermvars );
      assert( inbranchedvarindices[varid] );
      inbranchedvarindices[varid] = FALSE;
   }
#ifndef NDEBUG
   for (i = 0; i < orcdata->npermvars; ++i)
   {
      assert( inbranchedvarindices[i] == FALSE );
   }
#endif

   SCIPfreeCleanBufferArray(scip, &inbranchedvarindices);
   SCIPfreeBufferArray(scip, &branchedvarindices);

   return SCIP_OKAY;
}


/** applies orbital reduction on a symmetry group component using a two step mechanism
 *
 *  1. At the parent of our focus node (which is the current node, because we're not probing),
 *     compute the symmetry group just before branching. Then, for our branching variable x with variable y in its
 *     orbit, we mimic adding the constraint x >= y by variable bound propagations in this node.
 *
 *     In principle, this generalizes orbital branching in the binary case: propagation of x >= y yields
 *        1. In the 1-branch: 1 = x >= y is a tautology (since y is in {0, 1}). Nothing happens.
 *        0. In the 0-branch: 0 = x >= y implies y = 0. This is an exact description of orbital branching.
 *     REF: Ostrowski, James, et al. "Orbital branching." Mathematical Programming 126.1 (2011): 147-178.
 *
 *     (This only needs to be done once per node.)
 *
 *  2. At the focus node itself, compute the symmetry group.
 *     The symmetry group in this branch-and-bound tree node is a subgroup of the problem symmetry group
 *     as described in the function orbitalReductionGetSymmetryStabilizerSubgroup.
 *     For this symmetry subgroup, in each orbit, update the variable domains with the intersection of all variable
 *     domains in the orbit.
 *
 *     This generalizes orbital fixing in the binary case.
 *     REF: Margot 2002, Margot 2003, Orbital Branching, Ostrowski's PhD thesis.
 */
static
SCIP_RETCODE orbitalReductionPropagateComponent(
   SCIP*                 scip,               /**< SCIP data structure */
   ORCDATA*              orcdata,            /**< orbital reduction component data */
   SCIP_SHADOWTREE*      shadowtree,         /**< pointer to shadow tree */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is found */
   int*                  nred                /**< pointer to store the number of domain reductions */
   )
{
   assert( scip != NULL );
   assert( orcdata != NULL );
   assert( shadowtree != NULL );
   assert( infeasible != NULL );
   assert( nred != NULL );

   /* infeasible and nred are defined by the function that calls this function,
    * and this function only gets called if no infeasibility is found so far.
    */
   assert( !*infeasible );
   assert( *nred >= 0 );

   /* orbital reduction is only propagated when branching has started */
   assert( SCIPgetStage(scip) == SCIP_STAGE_SOLVING && SCIPgetNNodes(scip) > 1 );

   /* if this is the first call, identify the orbits for which symmetry is broken */
   if ( !orcdata->symmetrybrokencomputed )
   {
      SCIP_CALL( identifyOrbitalSymmetriesBroken(scip, orcdata) );
   }
   assert( orcdata->symmetrybrokencomputed );
   assert( orcdata->nsymbrokenvarids <= orcdata->npermvars );

   /* If symmetry is broken for all orbits, stop! */
   if ( orcdata->nsymbrokenvarids == orcdata->npermvars )
      return SCIP_OKAY;

   /* step 1 */
   SCIP_CALL( applyOrbitalBranchingPropagations(scip, orcdata, shadowtree, infeasible, nred) );
   if ( *infeasible )
      return SCIP_OKAY;

   /* step 2 */
   SCIP_CALL( applyOrbitalReductionPropagations(scip, orcdata, shadowtree, infeasible, nred) );
   if ( *infeasible )
      return SCIP_OKAY;

   return SCIP_OKAY;
}


/** adds component */
static
SCIP_RETCODE addComponent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA*  orbireddata,        /**< pointer to the orbital reduction data */
   SCIP_VAR**            permvars,           /**< variable array of the permutation */
   int                   npermvars,          /**< number of variables in that array */
   int**                 perms,              /**< permutations in the component */
   int                   nperms,             /**< number of permutations in the component */
   SCIP_Bool*            success             /**< to store whether the component is successfully added */
   )
{
   ORCDATA* orcdata;
   int i;
   int j;
   int p;
   int* origperm;
   int* newperm;
   int newidx;
   int newpermidx;

   assert( scip != NULL );
   assert( orbireddata != NULL );
   assert( permvars != NULL );
   assert( npermvars > 0 );
   assert( perms != NULL );
   assert( nperms > 0 );
   assert( success != NULL );

   *success = TRUE;
   SCIP_CALL( SCIPallocBlockMemory(scip, &orcdata) );

   /* correct indices by removing fixed points */

   /* determine the number of vars that are moved by the component, assign to orcdata->npermvars */
   orcdata->npermvars = 0;
   for (i = 0; i < npermvars; ++i)
   {
      /* is index i moved by any of the permutations in the component? */
      for (p = 0; p < nperms; ++p)
      {
         if ( perms[p][i] != i )
         {
            ++orcdata->npermvars;
            break;
         }
      }
   }

   /* do not support the setting where the component is empty */
   if ( orcdata->npermvars <= 0 )
   {
      SCIPfreeBlockMemory(scip, &orcdata);
      *success = FALSE;
      return SCIP_OKAY;
   }

   /* create index-corrected permvars array and the inverse */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &orcdata->permvars, orcdata->npermvars) );
   SCIP_CALL( SCIPhashmapCreate(&orcdata->permvarmap, SCIPblkmem(scip), orcdata->npermvars) );

   j = 0;
   for (i = 0; i < npermvars; ++i)
   {
      /* permvars array must be unique */
      assert( SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) permvars[i]) == INT_MAX );

      /* is index i moved by any of the permutations in the component? */
      for (p = 0; p < nperms; ++p)
      {
         if ( perms[p][i] != i )
         {
            /* var is moved by component; add, disable multiaggregation and capture */
            SCIP_CALL( SCIPcaptureVar(scip, permvars[i]) );
            orcdata->permvars[j] = permvars[i];
            SCIP_CALL( SCIPhashmapInsertInt(orcdata->permvarmap, (void*) permvars[i], j) );
            SCIP_CALL( SCIPmarkDoNotMultaggrVar(scip, permvars[i]) );
            ++j;
            break;
         }
      }
   }
   assert( j == orcdata->npermvars );

   /* allocate permutations */
   orcdata->nperms = nperms;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &orcdata->perms, nperms) );
   for (p = 0; p < nperms; ++p)
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &orcdata->perms[p], orcdata->npermvars) );
      origperm = perms[p];
      newperm = orcdata->perms[p];

      for (i = 0; i < npermvars; ++i)
      {
         newidx = SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) permvars[i]);
         if ( newidx >= orcdata->npermvars )
            continue;
         assert( newidx >= 0 );
         assert( newidx < orcdata->npermvars );
         assert( orcdata->permvars[newidx] == permvars[i] );
         newpermidx = SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) permvars[origperm[i]]);
         assert( newpermidx >= 0 );
         assert( newidx < orcdata->npermvars ); /* this is in the orbit of any permutation, so cannot be INT_MAX */
         assert( orcdata->permvars[newpermidx] == permvars[origperm[i]] );

         newperm[newidx] = newpermidx;
      }
   }

   /* global variable bounds */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &orcdata->globalvarlbs, orcdata->npermvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &orcdata->globalvarubs, orcdata->npermvars) );
   for (i = 0; i < orcdata->npermvars; ++i)
   {
      orcdata->globalvarlbs[i] = SCIPvarGetLbGlobal(orcdata->permvars[i]);
      orcdata->globalvarubs[i] = SCIPvarGetUbGlobal(orcdata->permvars[i]);
   }

   /* catch global variable bound change event */
   for (i = 0; i < orcdata->npermvars; ++i)
   {
      SCIP_CALL( SCIPcatchVarEvent(scip, orcdata->permvars[i], SCIP_EVENTTYPE_GLBCHANGED | SCIP_EVENTTYPE_GUBCHANGED,
         orbireddata->globalfixeventhdlr, (SCIP_EVENTDATA*) orcdata, NULL) );
   }

   /* lastnode field */
   orcdata->lastnode = NULL;

   /* symmetry computed field */
   orcdata->symmetrybrokencomputed = FALSE;
   orcdata->symbrokenvarids = NULL;
   orcdata->nsymbrokenvarids = -1;

   /* resize component array if needed */
   assert( orbireddata->ncomponents >= 0 );
   assert( (orbireddata->ncomponents == 0) == (orbireddata->componentdatas == NULL) );
   assert( orbireddata->ncomponents <= orbireddata->maxncomponents );
   if ( orbireddata->ncomponents == orbireddata->maxncomponents )
   {
      int newsize;

      newsize = SCIPcalcMemGrowSize(scip, orbireddata->ncomponents + 1);
      assert( newsize >= 0 );

      if ( orbireddata->ncomponents == 0 )
      {
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &orbireddata->componentdatas, newsize) );
      }
      else
      {
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &orbireddata->componentdatas,
            orbireddata->ncomponents, newsize) );
      }

      orbireddata->maxncomponents = newsize;
   }

   /* add component */
   assert( orbireddata->ncomponents < orbireddata->maxncomponents );
   orbireddata->componentdatas[orbireddata->ncomponents++] = orcdata;

   return SCIP_OKAY;
}


/** frees component */
static
SCIP_RETCODE freeComponent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA*  orbireddata,        /**< pointer to the orbital reduction data */
   ORCDATA**             orcdata             /**< pointer to component data */
   )
{
   int i;
   int p;

   assert( scip != NULL );
   assert( orbireddata != NULL );
   assert( orcdata != NULL );
   assert( *orcdata != NULL );
   assert( (*orcdata)->globalvarlbs != NULL );
   assert( (*orcdata)->globalvarubs != NULL );
   assert( (*orcdata)->nperms > 0 );
   assert( (*orcdata)->npermvars > 0 );
   assert( (*orcdata)->perms != NULL );
   assert( (*orcdata)->permvarmap != NULL );
   assert( (*orcdata)->permvars != NULL );
   assert( (*orcdata)->npermvars > 0 );

   assert( SCIPisTransformed(scip) );

   /* free symmetry broken information if it has been computed */
   if ( (*orcdata)->symmetrybrokencomputed )
   {
      assert( ((*orcdata)->nsymbrokenvarids == 0) == ((*orcdata)->symbrokenvarids == NULL) );
      SCIPfreeBlockMemoryArrayNull(scip, &(*orcdata)->symbrokenvarids, (*orcdata)->nsymbrokenvarids);
   }

   /* free global variable bound change event */
   if ( SCIPgetStage(scip) != SCIP_STAGE_FREE )
   {
      /* events at the freeing stage may not be dropped, because they are already getting dropped */
      for (i = (*orcdata)->npermvars - 1; i >= 0; --i)
      {
         SCIP_CALL( SCIPdropVarEvent(scip, (*orcdata)->permvars[i],
            SCIP_EVENTTYPE_GLBCHANGED | SCIP_EVENTTYPE_GUBCHANGED,
            orbireddata->globalfixeventhdlr, (SCIP_EVENTDATA*) (*orcdata), -1) );
      }
   }

   SCIPfreeBlockMemoryArray(scip, &(*orcdata)->globalvarubs, (*orcdata)->npermvars);
   SCIPfreeBlockMemoryArray(scip, &(*orcdata)->globalvarlbs, (*orcdata)->npermvars);

   for (p = (*orcdata)->nperms -1; p >= 0; --p)
   {
      SCIPfreeBlockMemoryArray(scip, &(*orcdata)->perms[p], (*orcdata)->npermvars);
   }
   SCIPfreeBlockMemoryArray(scip, &(*orcdata)->perms, (*orcdata)->nperms);

   /* release variables */
   for (i = 0; i < (*orcdata)->npermvars; ++i)
   {
      assert( (*orcdata)->permvars[i] != NULL );
      SCIP_CALL( SCIPreleaseVar(scip, &(*orcdata)->permvars[i]) );
   }

   SCIPhashmapFree(&(*orcdata)->permvarmap);
   SCIPfreeBlockMemoryArray(scip, &(*orcdata)->permvars, (*orcdata)->npermvars);

   SCIPfreeBlockMemory(scip, orcdata);

   return SCIP_OKAY;
}


/*
 * Event handler callback methods
 */

/** maintains global variable bound reductions found during presolving or at the root node */
static
SCIP_DECL_EVENTEXEC(eventExecGlobalBoundChange)
{
   ORCDATA* orcdata;
   SCIP_VAR* var;
   int varidx;

   assert( eventhdlr != NULL );
   assert( eventdata != NULL );
   assert( strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_SYMMETRY_NAME) == 0 );
   assert( event != NULL );

   /* only update the global bounds if branching has not started */
   if ( SCIPgetStage(scip) == SCIP_STAGE_SOLVING && SCIPgetNNodes(scip) > 1 )
      return SCIP_OKAY;

   orcdata = (ORCDATA*) eventdata;
   var = SCIPeventGetVar(event);
   assert( var != NULL );
   assert( SCIPvarIsTransformed(var) );
   assert( !orcdata->symmetrybrokencomputed );

   assert( orcdata->permvarmap != NULL );
   varidx = SCIPhashmapGetImageInt(orcdata->permvarmap, (void*) var);

   switch ( SCIPeventGetType(event) )
   {
   case SCIP_EVENTTYPE_GUBCHANGED:
      /* can assert with equality, because no arithmetic must be applied after inheriting the value of oldbound */
      assert( orcdata->globalvarubs[varidx] == SCIPeventGetOldbound(event) ); /*lint !e777 */
      orcdata->globalvarubs[varidx] = SCIPeventGetNewbound(event);
      break;
   case SCIP_EVENTTYPE_GLBCHANGED:
      assert( orcdata->globalvarlbs[varidx] == SCIPeventGetOldbound(event) ); /*lint !e777 */
      orcdata->globalvarlbs[varidx] = SCIPeventGetNewbound(event);
      break;
   default:
      SCIPABORT();
      return SCIP_ERROR;
   }

   return SCIP_OKAY;
}


/*
 * Interface methods
 */


/** prints orbital reduction data */
SCIP_RETCODE SCIPorbitalReductionGetStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA*  orbireddata,        /**< orbital reduction data structure */
   int*                  nred                /**< pointer to store the total number of reductions applied */
   )
{
   assert( scip != NULL );
   assert( orbireddata != NULL );

   *nred = orbireddata->nred;

   return SCIP_OKAY;
}

/** prints orbital reduction data */
SCIP_RETCODE SCIPorbitalReductionPrintStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA*  orbireddata         /**< orbital reduction data structure */
   )
{
   int i;

   assert( scip != NULL );
   assert( orbireddata != NULL );

   if ( orbireddata->ncomponents == 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "   orbital reduction:         no components\n");
      return SCIP_OKAY;
   }

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL,
      "   orbital reduction:       %4d components of sizes ", orbireddata->ncomponents);
   for (i = 0; i < orbireddata->ncomponents; ++i)
   {
      if ( i > 0 )
         SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, ", ");
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "%d", orbireddata->componentdatas[i]->nperms);
   }
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "\n");

   return SCIP_OKAY;
}


/** propagates orbital reduction */
SCIP_RETCODE SCIPorbitalReductionPropagate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA*  orbireddata,        /**< orbital reduction data structure */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is found */
   int*                  nred,               /**< pointer to store the number of domain reductions */
   SCIP_Bool*            didrun              /**< a global pointer maintaining if any symmetry propagator has run
                                              *   only set this to TRUE when a reduction is found, never set to FALSE */
   )
{
   ORCDATA* orcdata;
   SCIP_SHADOWTREE* shadowtree;
   int c;

   assert( scip != NULL );
   assert( orbireddata != NULL );
   assert( infeasible != NULL );
   assert( nred != NULL );
   assert( didrun != NULL );

   *infeasible = FALSE;
   *nred = 0;

   /* no components, no orbital reduction */
   assert( orbireddata->ncomponents >= 0 );
   if ( orbireddata->ncomponents == 0 )
      return SCIP_OKAY;

   /* do nothing if we are in a probing node */
   if ( SCIPinProbing(scip) )
      return SCIP_OKAY;

   /* do not run again in repropagation, since the path to the root might have changed */
   if ( SCIPinRepropagation(scip) )
      return SCIP_OKAY;

   assert( orbireddata->shadowtreeeventhdlr != NULL );
   shadowtree = SCIPgetShadowTree(orbireddata->shadowtreeeventhdlr);
   assert( shadowtree != NULL );

   for (c = 0; c < orbireddata->ncomponents; ++c)
   {
      orcdata = orbireddata->componentdatas[c];
      assert( orcdata != NULL );
      assert( orcdata->nperms > 0 );
      SCIP_CALL( orbitalReductionPropagateComponent(scip, orcdata, shadowtree, infeasible, nred) );

      /* a symmetry propagator has ran, so set didrun to TRUE */
      *didrun = TRUE;

      if ( *infeasible )
         break;
   }

   orbireddata->nred += *nred;

   return SCIP_OKAY;
}


/** adds component for orbital reduction */
SCIP_RETCODE SCIPorbitalReductionAddComponent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA*  orbireddata,        /**< orbital reduction data structure */
   SCIP_VAR**            permvars,           /**< variable array of the permutation */
   int                   npermvars,          /**< number of variables in that array */
   int**                 perms,              /**< permutations in the component */
   int                   nperms,             /**< number of permutations in the component */
   SCIP_Bool*            success             /**< to store whether the component is successfully added */
   )
{
   assert( scip != NULL );
   assert( orbireddata != NULL );
   assert( permvars != NULL );
   assert( npermvars > 0 );
   assert( perms != NULL );
   assert( nperms > 0 );
   assert( success != NULL );

   /* dynamic symmetry reductions cannot be performed on original problem */
   assert( SCIPisTransformed(scip) );

   SCIP_CALL( addComponent(scip, orbireddata, permvars, npermvars, perms, nperms, success) );

   return SCIP_OKAY;
}


/** resets orbital reduction data structure (clears all components) */
SCIP_RETCODE SCIPorbitalReductionReset(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA*  orbireddata         /**< orbital reduction data structure */
   )
{
   assert( scip != NULL );
   assert( orbireddata != NULL );
   assert( orbireddata->ncomponents >= 0 );
   assert( (orbireddata->ncomponents == 0) == (orbireddata->componentdatas == NULL) );
   assert( orbireddata->ncomponents <= orbireddata->maxncomponents );
   assert( orbireddata->shadowtreeeventhdlr != NULL );

   while ( orbireddata->ncomponents > 0 )
   {
      SCIP_CALL( freeComponent(scip, orbireddata, &(orbireddata->componentdatas[--orbireddata->ncomponents])) );
   }

   assert( orbireddata->ncomponents == 0 );
   SCIPfreeBlockMemoryArrayNull(scip, &orbireddata->componentdatas, orbireddata->maxncomponents);
   orbireddata->componentdatas = NULL;
   orbireddata->maxncomponents = 0;

   return SCIP_OKAY;
}


/** frees orbital reduction data */
SCIP_RETCODE SCIPorbitalReductionFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA** orbireddata         /**< orbital reduction data structure */
   )
{
   assert( scip != NULL );
   assert( orbireddata != NULL );
   assert( *orbireddata != NULL );

   SCIP_CALL( SCIPorbitalReductionReset(scip, *orbireddata) );

   SCIPfreeBlockMemory(scip, orbireddata);
   return SCIP_OKAY;
}


/** initializes structures needed for orbital reduction
 *
 *  This is only done exactly once.
 */
SCIP_RETCODE SCIPincludeOrbitalReduction(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ORBITALREDDATA** orbireddata,        /**< pointer to orbital reduction data structure to populate */
   SCIP_EVENTHDLR*       shadowtreeeventhdlr /**< pointer to the shadow tree eventhdlr */
   )
{
   assert( scip != NULL );
   assert( orbireddata != NULL );
   assert( shadowtreeeventhdlr != NULL );

   SCIP_CALL( SCIPcheckStage(scip, "SCIPincludeOrbitalReduction", TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
      FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPallocBlockMemory(scip, orbireddata) );

   (*orbireddata)->componentdatas = NULL;
   (*orbireddata)->ncomponents = 0;
   (*orbireddata)->maxncomponents = 0;
   (*orbireddata)->shadowtreeeventhdlr = shadowtreeeventhdlr;
   (*orbireddata)->nred = 0;

   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &(*orbireddata)->globalfixeventhdlr,
      EVENTHDLR_SYMMETRY_NAME, EVENTHDLR_SYMMETRY_DESC, eventExecGlobalBoundChange,
      (SCIP_EVENTHDLRDATA*) (*orbireddata)) );

   return SCIP_OKAY;
}
