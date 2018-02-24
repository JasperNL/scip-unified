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

/**@file   objbenderscut.h
 * @brief  C++ wrapper for Benders' decomposition cuts
 * @author Stephen J. Maher
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_OBJBENDERSCUT_H__
#define __SCIP_OBJBENDERSCUT_H__


#include <cassert>
#include <cstring>

#include "scip/scip.h"
#include "objscip/objcloneable.h"
#include "objscip/objbenders.h"

namespace scip
{

/**
 *  @brief C++ wrapper for Benders' decomposition cut
 *
 *  This class defines the interface for the Benders' decomposition cuts implemented in C++. Note that there is
 *  a pure virtual function (this must be implemented). This function is: benderscut_exec().
 *
 *  - \ref BENDERSCUT "Instructions for implementing a Benders' decomposition plugin"
 *  - \ref BENDERSDECOMPCUT "List of available Benders' decomposition plugins"
 *  - \ref type_benderscut.h "Corresponding C interface"
 */
class ObjBenderscut : public ObjCloneable
{
public:
   /*lint --e{1540}*/

   /** SCIP data structure */
   SCIP* scip_;

   /** name of the Benders' decomposition cut */
   char* scip_name_;

   /** description of the Benders' decomposition cut */
   char* scip_desc_;

   /** the priority of the Benders' decomposition cut */
   const int scip_priority_;

   /** is the Benders' decomposition cut generated from the LP relaxation of the subproblem */
   const SCIP_Bool scip_islpcut_;

   /** default constructor */
   ObjBenderscut(
      SCIP*              scip,               /**< SCIP data structure */
      const char*        name,               /**< name of Benders' decomposition */
      const char*        desc,               /**< description of Benders' decomposition */
      int                priority,           /**< priority of the Benders' decomposition */
      SCIP_Bool          islpcut             /**< is the cut generated from the LP relaxation */
      )
      : scip_(scip),
        scip_name_(0),
        scip_desc_(0),
        scip_priority_(priority),
        scip_islpcut_(islpcut)
   {
      /* the macro SCIPduplicateMemoryArray does not need the first argument: */
      SCIP_CALL_ABORT( SCIPduplicateMemoryArray(scip_, &scip_name_, name, std::strlen(name)+1) );
      SCIP_CALL_ABORT( SCIPduplicateMemoryArray(scip_, &scip_desc_, desc, std::strlen(desc)+1) );
   }

   /** destructor */
   virtual ~ObjBenderscut()
   {
      /* the macro SCIPfreeMemoryArray does not need the first argument: */
      /*lint --e{64}*/
      SCIPfreeMemoryArray(scip_, &scip_name_);
      SCIPfreeMemoryArray(scip_, &scip_desc_);
   }

   /** copy method for compression plugins (called when SCIP copies plugins)
    *
    *  @see SCIP_DECL_BENDERSCUTCOPY(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCUTCOPY(scip_copy)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** destructor of Benders' decomposition cuts to free user data (called when SCIP is exiting)
    *
    *  @see SCIP_DECL_BENDERSCUTFREE(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCUTFREE(scip_free)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** initialization method of Benders' decomposition cuts (called after problem was transformed)
    *
    *  @see SCIP_DECL_BENDERSCUTINIT(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCUTINIT(scip_init)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** deinitialization method of Benders' decomposition cuts (called before transformed problem is freed)
    *
    *  @see SCIP_DECL_BENDERSCUTEXIT(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCUTEXIT(scip_exit)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** solving process initialization method of Benders' decomposition cutsc (called when branch and bound process is about to begin)
    *
    *  @see SCIP_DECL_BENDERSCUTINITSOL(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCUTINITSOL(scip_initsol)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** solving process deinitialization method of Benders' decomposition cuts (called before branch and bound process data is freed)
    *
    *  This method is called before the branch and bound process is freed.
    *  The Benders' decomposition cuts should use this call to clean up its branch and bound data.
    *
    *  @see SCIP_DECL_BENDERSCUTEXITSOL(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCUTEXITSOL(scip_exitsol)
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** execution method of Benders' decomposition cuts technique
    *
    *  @see SCIP_DECL_BENDERSCUTEXEC(x) in @ref type_benders.h
    */
   virtual SCIP_DECL_BENDERSCUTEXEC(scip_exec) = 0;

};

} /* namespace scip */



/** creates the Benders' decomposition cut for the given Benders' decomposition cut object and includes it in SCIP
 *
 *  The method should be called in one of the following ways:
 *
 *   1. The user is resposible of deleting the object:
 *       SCIP_CALL( SCIPcreate(&scip) );
 *       ...
 *       MyBenderscut* mybenderscut = new MyBenderscut(...);
 *       SCIP_CALL( SCIPincludeObjBenderscut(scip, benders, &mybenderscut, FALSE) );
 *       ...
 *       SCIP_CALL( SCIPfree(&scip) );
 *       delete mybenderscut;    // delete benderscut AFTER SCIPfree() !
 *
 *   2. The object pointer is passed to SCIP and deleted by SCIP in the SCIPfree() call:
 *       SCIP_CALL( SCIPcreate(&scip) );
 *       ...
 *       SCIP_CALL( SCIPincludeObjBenderscut(scip, benders, new MyBenderscut(...), TRUE) );
 *       ...
 *       SCIP_CALL( SCIPfree(&scip) );  // destructor of MyBenderscut is called here
 */
EXTERN
SCIP_RETCODE SCIPincludeObjBenderscut(
   SCIP*                 scip,               /**< SCIP data structure */
   scip::ObjBenders*     objbenders,         /**< Benders' decomposition object */
   scip::ObjBenderscut*  objbenderscut,      /**< Benders' decomposition cut object */
   SCIP_Bool             deleteobject        /**< should the Benders' cut object be deleted when benderscut is freed? */
   );

/** returns the benderscut object of the given name, or 0 if not existing */
EXTERN
scip::ObjBenderscut* SCIPfindObjBenderscut(
   scip::ObjBenders*     objbenders,         /**< Benders' decomposition object */
   const char*           name                /**< name of Benders' decomposition cut */
   );

/** returns the benderscut object for the given constraint handler */
EXTERN
scip::ObjBenderscut* SCIPgetObjBenderscut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BENDERSCUT*      benderscut          /**< Benders' decomposition cut */
   );

#endif
