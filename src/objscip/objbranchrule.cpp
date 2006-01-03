/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2006 Tobias Achterberg                              */
/*                                                                           */
/*                  2002-2006 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License.             */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: objbranchrule.cpp,v 1.17 2006/01/03 12:22:40 bzfpfend Exp $"

/**@file   objbranchrule.cpp
 * @brief  C++ wrapper for branching rules
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>

#include "objbranchrule.h"




/*
 * Data structures
 */

/** branching rule data */
struct SCIP_BranchruleData
{
   scip::ObjBranchrule*  objbranchrule;      /**< branching rule object */
   SCIP_Bool             deleteobject;       /**< should the branching rule object be deleted when branching rule is freed? */
};




/*
 * Callback methods of branching rule
 */

/** destructor of branching rule to free user data (called when SCIP is exiting) */
static
SCIP_DECL_BRANCHFREE(branchFreeObj)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   assert(branchruledata->objbranchrule != NULL);

   /* call virtual method of branchrule object */
   SCIP_CALL( branchruledata->objbranchrule->scip_free(scip, branchrule) );

   /* free branchrule object */
   if( branchruledata->deleteobject )
      delete branchruledata->objbranchrule;

   /* free branchrule data */
   delete branchruledata;
   SCIPbranchruleSetData(branchrule, NULL);
   
   return SCIP_OKAY;
}


/** initialization method of branching rule (called after problem was transformed) */
static
SCIP_DECL_BRANCHINIT(branchInitObj)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   assert(branchruledata->objbranchrule != NULL);

   /* call virtual method of branchrule object */
   SCIP_CALL( branchruledata->objbranchrule->scip_init(scip, branchrule) );

   return SCIP_OKAY;
}


/** deinitialization method of branching rule (called before transformed problem is freed) */
static
SCIP_DECL_BRANCHEXIT(branchExitObj)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   assert(branchruledata->objbranchrule != NULL);

   /* call virtual method of branchrule object */
   SCIP_CALL( branchruledata->objbranchrule->scip_exit(scip, branchrule) );

   return SCIP_OKAY;
}


/** solving process initialization method of branching rule (called when branch and bound process is about to begin) */
static
SCIP_DECL_BRANCHINITSOL(branchInitsolObj)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   assert(branchruledata->objbranchrule != NULL);

   /* call virtual method of branchrule object */
   SCIP_CALL( branchruledata->objbranchrule->scip_initsol(scip, branchrule) );

   return SCIP_OKAY;
}


/** solving process deinitialization method of branching rule (called before branch and bound process data is freed) */
static
SCIP_DECL_BRANCHEXITSOL(branchExitsolObj)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   assert(branchruledata->objbranchrule != NULL);

   /* call virtual method of branchrule object */
   SCIP_CALL( branchruledata->objbranchrule->scip_exitsol(scip, branchrule) );

   return SCIP_OKAY;
}


/** branching execution method for fractional LP solutions */
static
SCIP_DECL_BRANCHEXECLP(branchExeclpObj)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   assert(branchruledata->objbranchrule != NULL);

   /* call virtual method of branchrule object */
   SCIP_CALL( branchruledata->objbranchrule->scip_execlp(scip, branchrule, allowaddcons, result) );

   return SCIP_OKAY;
}

/** branching execution method for not completely fixed pseudo solutions */
static
SCIP_DECL_BRANCHEXECPS(branchExecpsObj)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);
   assert(branchruledata->objbranchrule != NULL);

   /* call virtual method of branchrule object */
   SCIP_CALL( branchruledata->objbranchrule->scip_execps(scip, branchrule, allowaddcons, result) );

   return SCIP_OKAY;
}




/*
 * branching rule specific interface methods
 */

/** creates the branching rule for the given branching rule object and includes it in SCIP */
SCIP_RETCODE SCIPincludeObjBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   scip::ObjBranchrule*  objbranchrule,      /**< branching rule object */
   SCIP_Bool             deleteobject        /**< should the branching rule object be deleted when branching rule is freed? */
   )
{
   SCIP_BRANCHRULEDATA* branchruledata;

   /* create branching rule data */
   branchruledata = new SCIP_BRANCHRULEDATA;
   branchruledata->objbranchrule = objbranchrule;
   branchruledata->deleteobject = deleteobject;

   /* include branching rule */
   SCIP_CALL( SCIPincludeBranchrule(scip, objbranchrule->scip_name_, objbranchrule->scip_desc_, 
         objbranchrule->scip_priority_, objbranchrule->scip_maxdepth_, objbranchrule->scip_maxbounddist_,
         branchFreeObj, branchInitObj, branchExitObj, branchInitsolObj, branchExitsolObj,
         branchExeclpObj, branchExecpsObj,
         branchruledata) );

   return SCIP_OKAY;
}

/** returns the branchrule object of the given name, or NULL if not existing */
scip::ObjBranchrule* SCIPfindObjBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of branching rule */
   )
{
   SCIP_BRANCHRULE* branchrule;
   SCIP_BRANCHRULEDATA* branchruledata;

   branchrule = SCIPfindBranchrule(scip, name);
   if( branchrule == NULL )
      return NULL;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   return branchruledata->objbranchrule;
}
   
/** returns the branchrule object for the given branching rule */
scip::ObjBranchrule* SCIPgetObjBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule          /**< branching rule */
   )
{
   SCIP_BRANCHRULEDATA* branchruledata;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   return branchruledata->objbranchrule;
}
