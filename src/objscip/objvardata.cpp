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
#pragma ident "@(#) $Id: objvardata.cpp,v 1.8 2006/01/03 12:22:42 bzfpfend Exp $"

/**@file   objvardata.cpp
 * @brief  C++ wrapper for user variable data
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>

#include "objvardata.h"




/*
 * Data structures
 */

/** user variable data */
struct SCIP_VarData
{
   scip::ObjVardata*     objvardata;         /**< user variable data object */
   SCIP_Bool             deleteobject;       /**< should the user variable data object be deleted when variable is freed? */
};




/*
 * Callback methods of user variable data
 */

/** frees user data of original variable (called when the original variable is freed) */
static
SCIP_DECL_VARDELORIG(varDelorigObj)
{  /*lint --e{715}*/
   assert(vardata != NULL);
   assert(*vardata != NULL);
   assert((*vardata)->objvardata != NULL);

   /* call virtual method of vardata object */
   SCIP_CALL( (*vardata)->objvardata->scip_delorig(scip, var) );

   /* free vardata object */
   if( (*vardata)->deleteobject )
      delete (*vardata)->objvardata;

   /* free vardata data */
   delete *vardata;
   *vardata = NULL;
   
   return SCIP_OKAY;
}


/** creates user data of transformed variable by transforming the original user variable data
 *  (called after variable was transformed)
 */
static
SCIP_DECL_VARTRANS(varTransObj)
{  /*lint --e{715}*/
   scip::ObjVardata* objvardata;
   SCIP_Bool deleteobject;

   assert(sourcedata != NULL);
   assert(sourcedata->objvardata != NULL);
   assert(targetdata != NULL);
   assert(*targetdata == NULL);

   /* call virtual method of vardata object */
   SCIP_CALL( sourcedata->objvardata->scip_trans(scip, targetvar, &objvardata, &deleteobject) );

   /* create transformed user variable data */
   *targetdata = new SCIP_VARDATA;
   (*targetdata)->objvardata = objvardata;
   (*targetdata)->deleteobject = deleteobject;

   return SCIP_OKAY;
}


/** frees user data of transformed variable (called when the transformed variable is freed) */
static
SCIP_DECL_VARDELTRANS(varDeltransObj)
{  /*lint --e{715}*/
   assert(vardata != NULL);
   assert(*vardata != NULL);
   assert((*vardata)->objvardata != NULL);

   /* call virtual method of vardata object */
   SCIP_CALL( (*vardata)->objvardata->scip_deltrans(scip, var) );

   /* free vardata object */
   if( (*vardata)->deleteobject )
      delete (*vardata)->objvardata;

   /* free vardata data */
   delete *vardata;
   *vardata = NULL;
   
   return SCIP_OKAY;
}





/*
 * user variable data specific interface methods
 */

/** create and capture problem variable and associates the given variable data with the variable;
 *  if variable is of integral type, fractional bounds are automatically rounded
 */
SCIP_RETCODE SCIPcreateObjVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var,                /**< pointer to variable object */
   const char*           name,               /**< name of variable, or NULL for automatic name creation */
   SCIP_Real             lb,                 /**< lower bound of variable */
   SCIP_Real             ub,                 /**< upper bound of variable */
   SCIP_Real             obj,                /**< objective function value */
   SCIP_VARTYPE          vartype,            /**< type of variable */
   SCIP_Bool             initial,            /**< should var's column be present in the initial root LP? */
   SCIP_Bool             removeable,         /**< is var's column removeable from the LP (due to aging or cleanup)? */
   scip::ObjVardata*     objvardata,         /**< user variable data object */
   SCIP_Bool             deleteobject        /**< should the user variable data object be deleted when variable is freed? */
   )
{
   SCIP_VARDATA* vardata;

   /* create user variable data */
   vardata = new SCIP_VARDATA;
   vardata->objvardata = objvardata;
   vardata->deleteobject = deleteobject;

   /* create variable */
   SCIP_CALL( SCIPcreateVar(scip, var, name, lb, ub, obj, vartype, initial, removeable, 
         varDelorigObj, varTransObj, varDeltransObj, vardata) );

   return SCIP_OKAY;
}

/** gets user variable data object for given problem variable
 *  Warning! This method should only be called after a variable was created with SCIPcreateObjVar().
 *  Otherwise, a segmentation fault may arise, or an undefined pointer is returned.
 */
scip::ObjVardata* SCIPgetObjVardata(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_VARDATA* vardata;

   vardata = SCIPgetVarData(scip, var);
   assert(vardata != NULL);

   return vardata->objvardata;
}

