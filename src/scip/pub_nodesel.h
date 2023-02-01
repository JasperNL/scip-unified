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

/**@file   pub_nodesel.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for node selectors
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PUB_NODESEL_H__
#define __SCIP_PUB_NODESEL_H__


#include "scip/def.h"
#include "scip/type_nodesel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**@addtogroup PublicNodeSelectorMethods
 *
 * @{
 */

/** gets name of node selector */
SCIP_EXPORT
const char* SCIPnodeselGetName(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** gets description of node selector */
SCIP_EXPORT
const char* SCIPnodeselGetDesc(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** gets priority of node selector in standard mode */
SCIP_EXPORT
int SCIPnodeselGetStdPriority(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** gets priority of node selector in memory saving mode */
SCIP_EXPORT
int SCIPnodeselGetMemsavePriority(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** gets user data of node selector */
SCIP_EXPORT
SCIP_NODESELDATA* SCIPnodeselGetData(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** sets user data of node selector; user has to free old data in advance! */
SCIP_EXPORT
void SCIPnodeselSetData(
   SCIP_NODESEL*         nodesel,            /**< node selector */
   SCIP_NODESELDATA*     nodeseldata         /**< new node selector user data */
   );

/** is node selector initialized? */
SCIP_EXPORT
SCIP_Bool SCIPnodeselIsInitialized(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** gets time in seconds used in this node selector for setting up for next stages */
SCIP_EXPORT
SCIP_Real SCIPnodeselGetSetupTime(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** gets time in seconds used in this node selector */
SCIP_EXPORT
SCIP_Real SCIPnodeselGetTime(
   SCIP_NODESEL*         nodesel             /**< node selector */
   );

/** @} */

#ifdef __cplusplus
}
#endif

#endif
