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
/*  You should have received a copy of the ZIB Academic License.             */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   objbenders.h
 * @brief  C++ wrapper for Benders' decomposition
 * @author Stephen J. Maher
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_OBJBENDERS_H__
#define __SCIP_OBJBENDERS_H__


#include <cassert>
#include <cstring>

#include "scip/scip.h"
#include "objscip/objprobcloneable.h"

namespace scip
{

/**
 *  @brief C++ wrapper for Benders' decomposition plugins
 *
 *  This class defines the interface for the Benders' decomposition framework implemented in C++. Note that there
 *  are pure virtual functions (these have to be implemented). These functions are: benders_exec(), benders_createsub(),
 *  benders_getvar().
 *
 *  - \ref BENDERS "Instructions for implementing a Benders' decomposition plugin"
 *  - \ref BENDERSDECOMP "List of available Benders' decomposition plugins"
 *  - \ref type_benders.h "Corresponding C interface"
 */
class ObjBenders : public ObjProbCloneable
{
public:
   /*lint --e{1540}*/

   /** SCIP data structure */
   SCIP* scip_;

   /** Benders' decomposition data structure */
   SCIP_BENDERS* benders_;

   /** name of the Benders' decomposition */
   char* scip_name_;

   /** description of the Benders' decomposition */
   char* scip_desc_;

   /** the priority of the Benders' decomposition */
   const int scip_priority_;

   /** should cuts be generated from the LP solution */
   const SCIP_Bool scip_cutlp_;

   /** should cuts be generated from the pseudo solution */
   const SCIP_Bool scip_cutpseudo_;

   /** should cuts be generated from the relaxation solution */
   const SCIP_Bool scip_cutrelax_;

   /** should this Benders' decomposition share the auxiliary variables from the highest priority Benders? */
   const SCIP_Bool scip_shareauxvars_;

   /** default constructor */
   ObjBenders(
      SCIP*              scip,               /**< SCIP data structure */
      const char*        name,               /**< name of Benders' decomposition */
      const char*        desc,               /**< description of Benders' decomposition */
      int                priority,           /**< priority of the Benders' decomposition */
      SCIP_Bool          cutlp,              /**< should Benders' cuts be generated for LP solutions */
      SCIP_Bool          cutpseudo,          /**< should Benders' cuts be generated for pseudo solutions */
      SCIP_Bool          cutrelax,           /**< should Benders' cuts be generated for relaxation solutions */
      SCIP_Bool          shareauxvars        /**< should this Benders' use the highest priority Benders' aux vars */
      )
      : scip_(scip),
        benders_(0),
        scip_name_(0),
        scip_desc_(0),
        scip_priority_(priority),
        scip_cutlp_(cutlp),
        scip_cutpseudo_(cutpseudo),
        scip_cutrelax_(cutrelax),
        scip_shareauxvars_(shareauxvars)
   {
      /* the macro SCIPduplicateMemoryArray does not need the first argument: */
      SCIP_CALL_ABORT( SCIPduplicateMemoryArray(scip_, &scip_name_, name, std::strlen(name)+1) );
      SCIP_CALL_ABORT( SCIPduplicateMemoryArray(scip_, &scip_desc_, desc, std::strlen(desc)+1) );
   }

   /** destructor */
   virtual ~ObjBenders()
   {
      /* the macro SCIPfreeMemoryArray does not need the first argument: */
      /*lint --e{64}*/
      SCIPfreeMemoryArray(scip_, &scip_name_);
      SCIPfreeMemoryArray(scip_, &scip_desc_);
   }

   /** copy method for benders plugins (called when SCIP copies plugins)
    *
    *  @see SCIP_DECL_BENDERSCOPY(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCOPY(scip_copy)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** destructor of variable benders to free user data (called when SCIP is exiting)
    *
    *  @see SCIP_DECL_BENDERSFREE(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSFREE(scip_free)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** initialization method of variable benders (called after problem was transformed and benders is active)
    *
    *  @see SCIP_DECL_BENDERSINIT(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSINIT(scip_init)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** deinitialization method of variable benders (called before transformed problem is freed and benders is active)
    *
    *  @see SCIP_DECL_BENDERSEXIT(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSEXIT(scip_exit)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** presolving initialization method of constraint handler (called when presolving is about to begin)
    *
    *  @see SCIP_DECL_BENDERSINITPRE(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSINITPRE(scip_initpre)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** presolving deinitialization method of constraint handler (called after presolving has been finished)
    *
    *  @see SCIP_DECL_BENDERSEXITPRE(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSEXITPRE(scip_exitpre)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** solving process initialization method of variable benders (called when branch and bound process is about to begin)
    *
    *  @see SCIP_DECL_BENDERSINITSOL(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSINITSOL(scip_initsol)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** solving process deinitialization method of variable benders (called before branch and bound process data is freed)
    *
    *  @see SCIP_DECL_BENDERSEXITSOL(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSEXITSOL(scip_exitsol)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** the method for creating the Benders' decomposition subproblem. This method is called during the initialisation stage
    *  (after the master problem was transformed)
    *
    *   @see SCIP_DECL_BENDERSCREATESUB(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCREATESUB(scip_createsub) = 0;

   /** called before the subproblem solving loop for Benders' decomposition. The pre subproblem solve function gives the
    *  user an oppportunity to perform any global set up for the Benders' decomposition.
    *
    *   @see SCIP_DECL_BENDERSPRESUBSOLVE(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSPRESUBSOLVE(scip_presubsolve) = 0;

   /** the solving method for a single Benders' decomposition subproblem. The solving methods are separated so that they
    *  can be called in parallel.
    *
    *   @see SCIP_DECL_BENDERSSOLVESUB(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSSOLVESUB(scip_solvesub)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** the post-solve method for Benders' decomposition. The post-solve method is called after the subproblems have
    * been solved but before they are freed.
    *  @see SCIP_DECL_BENDERSPOSTSOLVE(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSPOSTSOLVE(scip_postsolve)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** frees the subproblem so that it can be resolved in the next iteration. In the SCIP case, this involves freeing the
    *  transformed problem using SCIPfreeTransform()
    *
    *   @see SCIP_DECL_BENDERSFREESUB(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSFREESUB(scip_freesub)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** the variable mapping from the subproblem to the master problem.
    *
    *  @see SCIP_DECL_BENDERSGETVAR(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSGETVAR(scip_getvar) = 0;

};

} /* namespace scip */



/** creates the Benders' decomposition for the given Benders' decomposition object and includes it in SCIP
 *
 *  The method should be called in one of the following ways:
 *
 *   1. The user is resposible of deleting the object:
 *       SCIP_CALL( SCIPcreate(&scip) );
 *       ...
 *       MyBenders* mybenders = new MyBenders(...);
 *       SCIP_CALL( SCIPincludeObjBenders(scip, &mybenders, FALSE) );
 *       ...
 *       SCIP_CALL( SCIPfree(&scip) );
 *       delete mybenders;    // delete benders AFTER SCIPfree() !
 *
 *   2. The object pointer is passed to SCIP and deleted by SCIP in the SCIPfree() call:
 *       SCIP_CALL( SCIPcreate(&scip) );
 *       ...
 *       SCIP_CALL( SCIPincludeObjBenders(scip, new MyBenders(...), TRUE) );
 *       ...
 *       SCIP_CALL( SCIPfree(&scip) );  // destructor of MyBenders is called here
 */
EXTERN
SCIP_RETCODE SCIPincludeObjBenders(
   SCIP*                 scip,               /**< SCIP data structure */
   scip::ObjBenders*     objbenders,         /**< Benders' decomposition object */
   SCIP_Bool             deleteobject        /**< should the Benders' decomposition object be deleted when benders is freed? */
   );

/** returns the benders object of the given name, or 0 if not existing */
EXTERN
scip::ObjBenders* SCIPfindObjBenders(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of Benders' decomposition */
   );

/** returns the benders object for the given constraint handler */
EXTERN
scip::ObjBenders* SCIPgetObjBenders(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BENDERS*         benders             /**< Benders' decomposition */
   );

#endif
