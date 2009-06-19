/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2009 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: paramset.h,v 1.24.2.1 2009/06/19 07:53:46 bzfwolte Exp $"

/**@file   paramset.h
 * @brief  internal methods for handling parameter settings
 * @author Tobias Achterberg
 * @author Timo Berthold
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PARAMSET_H__
#define __SCIP_PARAMSET_H__


#include "scip/def.h"
#include "blockmemshell/memory.h"
#include "scip/type_retcode.h"
#include "scip/type_paramset.h"
#include "scip/pub_paramset.h"
#include "scip/pub_misc.h"



/** creates parameter set */
extern
SCIP_RETCODE SCIPparamsetCreate(
   SCIP_PARAMSET**       paramset,           /**< pointer to store the parameter set */
   BMS_BLKMEM*           blkmem              /**< block memory */
   );

/** frees parameter set */
extern
void SCIPparamsetFree(
   SCIP_PARAMSET**       paramset,           /**< pointer to the parameter set */
   BMS_BLKMEM*           blkmem              /**< block memory */
   );

/** creates a bool parameter, sets it to its default value, and adds it to the parameter set */
extern
SCIP_RETCODE SCIPparamsetAddBool(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Bool*            valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Bool             defaultvalue,       /**< default value of the parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a int parameter, sets it to its default value, and adds it to the parameter set */
extern
SCIP_RETCODE SCIPparamsetAddInt(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   int*                  valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   int                   defaultvalue,       /**< default value of the parameter */
   int                   minvalue,           /**< minimum value for parameter */
   int                   maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a SCIP_Longint parameter, sets it to its default value, and adds it to the parameter set */
extern
SCIP_RETCODE SCIPparamsetAddLongint(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Longint*         valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Longint          defaultvalue,       /**< default value of the parameter */
   SCIP_Longint          minvalue,           /**< minimum value for parameter */
   SCIP_Longint          maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a SCIP_Real parameter, sets it to its default value, and adds it to the parameter set */
extern
SCIP_RETCODE SCIPparamsetAddReal(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Real*            valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Real             defaultvalue,       /**< default value of the parameter */
   SCIP_Real             minvalue,           /**< minimum value for parameter */
   SCIP_Real             maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a char parameter, sets it to its default value, and adds it to the parameter set */
extern
SCIP_RETCODE SCIPparamsetAddChar(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   char*                 valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   char                  defaultvalue,       /**< default value of the parameter */
   const char*           allowedvalues,      /**< array with possible parameter values, or NULL if not restricted */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** creates a string parameter, sets it to its default value, and adds it to the parameter set */
extern
SCIP_RETCODE SCIPparamsetAddString(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   char**                valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   const char*           defaultvalue,       /**< default value of the parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   );

/** gets the value of an existing SCIP_Bool parameter */
extern
SCIP_RETCODE SCIPparamsetGetBool(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   const char*           name,               /**< name of the parameter */
   SCIP_Bool*            value               /**< pointer to store the parameter */
   );

/** gets the value of an existing int parameter */
extern
SCIP_RETCODE SCIPparamsetGetInt(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   const char*           name,               /**< name of the parameter */
   int*                  value               /**< pointer to store the parameter */
   );

/** gets the value of an existing SCIP_Longint parameter */
extern
SCIP_RETCODE SCIPparamsetGetLongint(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   const char*           name,               /**< name of the parameter */
   SCIP_Longint*         value               /**< pointer to store the parameter */
   );

/** gets the value of an existing SCIP_Real parameter */
extern
SCIP_RETCODE SCIPparamsetGetReal(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   const char*           name,               /**< name of the parameter */
   SCIP_Real*            value               /**< pointer to store the parameter */
   );

/** gets the value of an existing char parameter */
extern
SCIP_RETCODE SCIPparamsetGetChar(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   const char*           name,               /**< name of the parameter */
   char*                 value               /**< pointer to store the parameter */
   );

/** gets the value of an existing string parameter */
extern
SCIP_RETCODE SCIPparamsetGetString(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   const char*           name,               /**< name of the parameter */
   char**                value               /**< pointer to store the parameter */
   );

/** changes the value of an existing SCIP_Bool parameter */
extern
SCIP_RETCODE SCIPparamsetSetBool(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP_SET*             set,                /**< global SCIP settings */
   const char*           name,               /**< name of the parameter */
   SCIP_Bool             value               /**< new value of the parameter */
   );

/** changes the value of an existing int parameter */
extern
SCIP_RETCODE SCIPparamsetSetInt(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP_SET*             set,                /**< global SCIP settings */
   const char*           name,               /**< name of the parameter */
   int                   value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Longint parameter */
extern
SCIP_RETCODE SCIPparamsetSetLongint(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP_SET*             set,                /**< global SCIP settings */
   const char*           name,               /**< name of the parameter */
   SCIP_Longint          value               /**< new value of the parameter */
   );

/** changes the value of an existing SCIP_Real parameter */
extern
SCIP_RETCODE SCIPparamsetSetReal(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP_SET*             set,                /**< global SCIP settings */
   const char*           name,               /**< name of the parameter */
   SCIP_Real             value               /**< new value of the parameter */
   );

/** changes the value of an existing char parameter */
extern
SCIP_RETCODE SCIPparamsetSetChar(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP_SET*             set,                /**< global SCIP settings */
   const char*           name,               /**< name of the parameter */
   char                  value               /**< new value of the parameter */
   );

/** changes the value of an existing string parameter */
extern
SCIP_RETCODE SCIPparamsetSetString(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP_SET*             set,                /**< global SCIP settings */
   const char*           name,               /**< name of the parameter */
   const char*           value               /**< new value of the parameter */
   );

/** reads parameters from a file */
SCIP_RETCODE SCIPparamsetRead(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP_SET*             set,                /**< global SCIP settings */
   const char*           filename            /**< file name */
   );

/** writes all parameters in the parameter set to a file */
SCIP_RETCODE SCIPparamsetWrite(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   const char*           filename,           /**< file name, or NULL for stdout */
   SCIP_Bool             comments,           /**< should parameter descriptions be written as comments? */
   SCIP_Bool             onlychanged         /**< should only the parameters been written, that are changed from default? */
   );

/** installs default values for all parameters */
extern
SCIP_RETCODE SCIPparamsetSetToDefault(
   SCIP_PARAMSET*        paramset,           /**< parameter set */
   SCIP*                 scip                /**< SCIP data structure, or NULL if paramchgd method should not be called */   
   );

/** returns the array of parameters */
extern
SCIP_PARAM** SCIPparamsetGetParams(
   SCIP_PARAMSET*        paramset            /**< parameter set */
   );

/** returns the number of parameters in the parameter set */
extern
int SCIPparamsetGetNParams(
   SCIP_PARAMSET*        paramset            /**< parameter set */
   );

#endif
