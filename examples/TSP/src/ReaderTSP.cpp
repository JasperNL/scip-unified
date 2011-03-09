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
/*  You should have received a copy of the ZIB Academic License.             */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   ReaderTSP.cpp
 * @brief  C++ file reader for TSP data files
 * @author Timo Berthold
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "objscip/objscip.h"

#include "scip/cons_linear.h"
#include <math.h>

#include "ReaderTSP.h"
#include "ProbDataTSP.h"
#include "ConshdlrSubtour.h"
#include "GomoryHuTree.h"

using namespace tsp;
using namespace scip;
using namespace std;

#define NINT(x) (floor(x+0.5))
#define FRAC(x) (x-floor(x))

/** parses the node list */ 
void ReaderTSP::getNodesFromFile(
   std::ifstream&     filedata,           /**< filestream containing the data to extract */
   double*            x_coords,           /**< double array to be filled with the x-coordinates of the nodes */
   double*            y_coords,           /**< same for y-coordinates */
   GRAPH*             graph               /**< the graph which is to be generated by the nodes */
   )
{
   int i = 0;
   int nodenumber;
   GRAPHNODE* node = &(graph->nodes[0]);

   // extract every node out of the filestream
   while ( i < graph->nnodes && !filedata.eof() )
   {
      filedata >> nodenumber >> x_coords[i] >> y_coords[i];

      // assign everything 
      node->id = i;
      if( nodenumber-1 != i)
         cout<<"warning: nodenumber <" <<nodenumber<<"> does not match its index in node list <"<<i+1
             <<">. Node will get number "<<i+1<<" when naming variables and constraints!"<<endl;
      node->x = x_coords[i];
      node->y = y_coords[i];
      node->first_edge = NULL; 
      node++; 
      i++;
   }
   assert( i == graph->nnodes );
}

/** adds a variable to both halfedges and captures it for usage in the graph */
SCIP_RETCODE ReaderTSP::addVarToEdges(
   SCIP*                 scip,               /**< SCIP data structure */
   GRAPHEDGE*            edge,               /**< an edge of the graph */
   SCIP_VAR*             var                 /**< variable corresponding to that edge */
   )
{
   assert(scip != NULL);
   assert(edge != NULL);
   assert(var != NULL);

   /* add variable to forward edge and capture it for usage in graph */
   edge->var = var;
   SCIP_CALL( SCIPcaptureVar(scip, edge->var) );

   /* two parallel halfedges have the same variable,
    * add variable to backward edge and capture it for usage in graph */
   edge->back->var = edge->var;
   SCIP_CALL( SCIPcaptureVar(scip, edge->back->var) );

   return SCIP_OKAY;
}

/** method asserting, that the file has had the correct format and everything was set correctly */
bool ReaderTSP::checkValid(
   GRAPH*             graph,              /**< the constructed graph, schould not be NULL */ 
   std::string        name,               /**< the name of the file */
   std::string        type,               /**< the type of the problem, should be "TSP" */
   std::string        edgeweighttype,     /**< type of the edgeweights, should be "EUC_2D", "MAX_2D", "MAN_2D", 
                                           *   "ATT", or "GEO" */
   int                nnodes              /**< dimension of the problem, should at least be one */
   )
{   
   // if something seems to be strange, it will be reported, that file was not valid
   if( nnodes < 1 )
   {
      cout << "parse error in file <" << name << "> dimension should be greater than 0"<< endl ;
      return false;
   }
   if (type != "TSP" )
   {  
      cout << "parse error in file <" << name << "> type should be TSP" << endl;
      return false;
   }
   if ( !( edgeweighttype == "EUC_2D" || edgeweighttype == "MAX_2D" || edgeweighttype == "MAN_2D" 
         || edgeweighttype == "GEO" || edgeweighttype == "ATT") )
   {
      cout << "parse error in file <" << name 
           << "> unknown weight type, should be EUC_2D, MAX_2D, MAN_2D, ATT, or GEO" << endl;
      return false;
   }
   if( graph == NULL)
   {
      cout << "error while reading file <" << name << ">, graph is uninitialized. "; 
      cout << "Probably NODE_COORD_SECTION is missing" << endl;
      return false;
   }
   return true;
}


/** destructor of file reader to free user data (called when SCIP is exiting) */
SCIP_RETCODE ReaderTSP::scip_free(
   SCIP*              scip,               /**< SCIP data structure */
   SCIP_READER*       reader              /**< the file reader itself */
   )
{
   return SCIP_OKAY;
}

/** problem reading method of reader
 *
 *  possible return values for *result:
 *  - SCIP_SUCCESS    : the reader read the file correctly and created an appropritate problem
 *  - SCIP_DIDNOTRUN  : the reader is not responsible for given input file
 *
 *  If the reader detected an error in the input file, it should return with RETCODE SCIP_READERR or SCIP_NOFILE.
 */
SCIP_RETCODE ReaderTSP::scip_read(
   SCIP*              scip,               /**< SCIP data structure */
   SCIP_READER*       reader,             /**< the file reader itself */
   const char*        filename,           /**< full path and name of file to read, or NULL if stdin should be used */
   SCIP_RESULT*       result              /**< pointer to store the result of the file reading call */
   )
{
   SCIP_RETCODE retcode;

   GRAPH* graph = NULL;
   GRAPHNODE* node;
   GRAPHNODE* nodestart;             // the two incident nodes of an edge
   GRAPHNODE* nodeend;
   GRAPHEDGE* edgeforw;              // two converse halfedges
   GRAPHEDGE* edgebackw;
   GRAPHEDGE* edge;

   double*  x_coords = NULL;                 // arrays of coordinates of the nodes
   double*  y_coords = NULL;

#ifdef SCIP_DEBUG
   double** weights = NULL;
#endif
   
   double x;                         // concrete coordinates
   double y;

   int nnodes = 0;
   int nedges = 0;
   int i;
   int j;

   string name = "MY_OWN_LITTLE_TSP";
   string token;
   string type = "TSP";
   string edgeweighttype = "EUC_2D";

   retcode = SCIP_OKAY;
   *result = SCIP_DIDNOTRUN;

   // open the file
   ifstream filedata(filename);
   if( !filedata )
      return SCIP_READERROR;
   filedata.clear();

   // read the first lines of information
   filedata >> token;
   while( !filedata.eof() )
   {
      if( token == "NAME:" )
         filedata >> name;
      else if( token == "NAME" )
         filedata >> token >> name;
      else if( token == "TYPE:" )
         filedata >> type;
      else if( token == "TYPE" )
         filedata >> token >> type;
      else if( token == "DIMENSION:" )
      {
         filedata >> nnodes;
         nedges = nnodes * (nnodes-1);
      }
      else if( token == "DIMENSION" )
      {
         filedata >> token >> nnodes;
         nedges = nnodes * (nnodes-1);
      }
      else if( token == "EDGE_WEIGHT_TYPE:" )
         filedata >> edgeweighttype;
      else if( token == "EDGE_WEIGHT_TYPE" )
         filedata >> token >> edgeweighttype;
      else if( token == "NODE_COORD_SECTION" || token == "DISPLAY_DATA_SECTION" )
      {
         // there should be some nodes to construct a graph
         if( nnodes < 1 )
         {
            retcode = SCIP_READERROR;
            break;
         }
         // the graph is created and filled with nodes 
         else if( create_graph(nnodes, nedges, &graph) )
         {
            assert(x_coords == NULL);
            assert(y_coords == NULL);

            x_coords = new double[nnodes];
            y_coords = new double[nnodes];
            getNodesFromFile(filedata, x_coords, y_coords, graph);
         }
         else
         {
            retcode = SCIP_NOMEMORY;
            break;
         }
      }  
      else if( token == "COMMENT:" || token == "COMMENT" || 
         token == "DISPLAY_DATA_TYPE" || token == "DISPLAY_DATA_TYPE:" )
         getline( filedata, token ); 
      else if( token == "EOF" )
         break;
      else if( token == "" )
         ;
      else
      {
         cout << "parse error in file <" << name << "> unknown keyword <" << token << ">" << endl;
         return SCIP_READERROR;
      }
      filedata >> token;
   }// finished parsing the input
   
   // check whether the input data was valid
   if( !checkValid(graph, name, type, edgeweighttype, nnodes) )
      retcode = SCIP_READERROR;

   if( retcode == SCIP_OKAY )
   {
      edgeforw = &( graph->edges[0] ); 
      edgebackw= &( graph->edges[nedges/2] );

#ifdef SCIP_DEBUG
      weights = new double* [nnodes];
      for( i = 0; i < nnodes; ++i )
         weights[i] = new double[nnodes];
#endif

      // construct all edges in a complete digraph
      for( i = 0; i < nnodes; i++ )
      {
         nodestart = &graph->nodes[i];
         for( j = i+1; j < nnodes; j++ )
         {
            nodeend = &graph->nodes[j];

            // construct two 'parallel' halfedges
            edgeforw->adjac = nodeend;
            edgebackw->adjac = nodestart;
            edgeforw->back = edgebackw;
            edgebackw->back = edgeforw;

            // calculate the Euclidean / Manhattan / Maximum distance of the two nodes
            x = x_coords[(*nodestart).id] -  x_coords[(*nodeend).id];
            y = y_coords[(*nodestart).id] -  y_coords[(*nodeend).id];
            if( edgeweighttype == "EUC_2D")
               edgeforw->length = sqrt( x*x + y*y );
            else if( edgeweighttype == "MAX_2D")
               edgeforw->length = max( ABS(x), ABS(y) );
            else if( edgeweighttype == "MAN_2D")
               edgeforw->length = ABS(x) + ABS(y);
            else if( edgeweighttype == "ATT")
               edgeforw->length = ceil( sqrt( (x*x+y*y)/10.0 ) ); 
            else if( edgeweighttype == "GEO")
            {
               const double pi =  3.141592653589793;
               double rads[4];
               double coords[4];
               double degs[4];
               double mins[4];
               double euler[3];
               int k;

               coords[0] = x_coords[(*nodestart).id];
               coords[1] = y_coords[(*nodestart).id];
               coords[2] = x_coords[(*nodeend).id];
               coords[3] = y_coords[(*nodeend).id];

               for( k = 0; k < 4; k++ )
               {
                  degs[k] = coords[k] >= 0 ? floor(coords[k]) : ceil(coords[k]);
                  mins[k] = coords[k] - degs[k];
                  rads[k] = pi*(degs[k]+5.0*mins[k]/3.0)/180.0;
               }
               
               euler[0] = cos(rads[1]-rads[3]);
               euler[1] = cos(rads[0]-rads[2]);
               euler[2] = cos(rads[0]+rads[2]);
               edgeforw->length = floor(6378.388 * acos(0.5*((1.0+euler[0])*euler[1]-(1.0-euler[0])*euler[2]))+1.0);
            }          
            
            // in TSP community, it is common practice to round lengths to next integer
            if( round_lengths_ )
               edgeforw->length = NINT(edgeforw->length);
            
            edgebackw->length = edgeforw->length;
#ifdef SCIP_DEBUG
            weights[i][j] = edgeforw->length;
            weights[j][i] = edgebackw->length;
#endif
   
            // insert one of the halfedges into the edge list of the node
            if (nodestart->first_edge == NULL)
            {
               nodestart->first_edge = edgeforw;
               nodestart->first_edge->next = NULL;
            }
            else
            {
               edgeforw->next = nodestart->first_edge;
               nodestart->first_edge = edgeforw;
            }
                   
            // dito
            if (nodeend->first_edge == NULL)
            {
               nodeend->first_edge = edgebackw;
               nodeend->first_edge->next = NULL;
            }
            else
            {
               edgebackw->next = nodeend->first_edge;
               nodeend->first_edge = edgebackw;
            }           
                     
            edgeforw++;
            edgebackw++;            
         }
      }
   }

   delete[] y_coords;
   delete[] x_coords;
      
   if( retcode != SCIP_OKAY )
   {
#ifdef SCIP_DEBUG
      if( weights != NULL )
      {
         for( i = 0; i < nnodes; i++ )
         {    
            delete[] weights[i];
         }
         delete[] weights;
      }
#endif
      return retcode;
   }

#ifdef SCIP_DEBUG
   printf("Matrix:\n");
   for( i = 0; i < nnodes; i++ )
   {    
      for( j = 0; j < nnodes; j++ )
         printf(" %4.0f ",weights[i][j]);
      printf("\n");
      delete[] weights[i];
   }
   delete[] weights;
#endif

   // create the problem's data structure
   SCIP_CALL( SCIPcreateObjProb(scip, name.c_str(), new ProbDataTSP(graph), TRUE) );

   // add variables to problem and link them for parallel halfedges
   for( i = 0; i < nedges/2; i++ )
   {
      SCIP_VAR* var;

      stringstream varname;
      edge = &graph->edges[i];

      // the variable is named after the two nodes connected by the edge it represents
      varname << "x_e_" << edge->back->adjac->id+1 << "-" << edge->adjac->id+1;
      SCIP_CALL( SCIPcreateVar(scip, &var, varname.str().c_str(), 0.0, 1.0, edge->length,
            SCIP_VARTYPE_BINARY, TRUE, FALSE, NULL, NULL, NULL, NULL, NULL) );

      /* add variable to SCIP and to the graph */
      SCIP_CALL( SCIPaddVar(scip, var) );
      SCIP_CALL( addVarToEdges(scip, edge, var) );

      /* release variable for the reader. */
      SCIP_CALL( SCIPreleaseVar(scip, &var) );      

   }

   // add all n node degree constraints
   if( nnodes >= 2 )
   {
      for( i = 0, node = &(graph->nodes[0]); i < nnodes; i++, node++ )
      {
         SCIP_CONS* cons;
         stringstream consname;
         consname << "deg_con_v" << node->id+1;
         
         // a new degree constraint is created, named after a node
         SCIP_CALL( SCIPcreateConsLinear(scip, &cons, consname.str().c_str(), 0, NULL, NULL, 2.0, 2.0, 
               TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );  

         edge = node->first_edge;
         // sum up the values of all adjacent edges 
         while( edge != NULL )
         {
            SCIP_CALL( SCIPaddCoefLinear(scip, cons, edge->var, 1.0) );
            edge = edge->next;
         }
             
         // add the constraint to SCIP
         SCIP_CALL( SCIPaddCons(scip, cons) );
         SCIP_CALL( SCIPreleaseCons(scip, &cons) );
      }
   }

   // last, we need a constraint forbidding subtours
   SCIP_CONS* cons;
   SCIP_CALL( SCIPcreateConsSubtour(scip, &cons, "subtour", graph, 
         FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE ) ); /* aus cons_subtour.h; eigener Constraint Handler */
   SCIP_CALL( SCIPaddCons(scip, cons) );
   SCIP_CALL( SCIPreleaseCons(scip, &cons) );
 
   release_graph(&graph);
   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}

/** problem writing method of reader; NOTE: if the parameter "genericnames" is TRUE, then
 *  SCIP already set all variable and constraint names to generic names; therefore, this
 *  method should always use SCIPvarGetName() and SCIPconsGetName(); 
 *
 *  possible return values for *result:
 *  - SCIP_SUCCESS    : the reader read the file correctly and created an appropritate problem
 *  - SCIP_DIDNOTRUN  : the reader is not responsible for given input file
 *
 *  If the reader detected an error in the writing to the file stream, it should return
 *  with RETCODE SCIP_WRITEERROR.
 */
SCIP_RETCODE ReaderTSP::scip_write(
   SCIP*              scip,               /**< SCIP data structure */
   SCIP_READER*       reader,             /**< the file reader itself */
   FILE*              file,               /**< output file, or NULL if standard output should be used */
   const char*        name,               /**< problem name */
   SCIP_PROBDATA*     probdata,           /**< user problem data set by the reader */
   SCIP_Bool          transformed,        /**< TRUE iff problem is the transformed problem */

   SCIP_OBJSENSE      objsense,           /**< objective sense */
   SCIP_Real          objscale,           /**< scalar applied to objective function; external objective value is
                                           *   extobj = objsense * objscale * (intobj + objoffset) */
   SCIP_Real          objoffset,          /**< objective offset from bound shifting and fixing */
   SCIP_VAR**         vars,               /**< array with active variables ordered binary, integer, implicit, 
                                           *   continuous */
   int                nvars,              /**< number of mutable variables in the problem */
   int                nbinvars,           /**< number of binary variables */
   int                nintvars,           /**< number of general integer variables */
   int                nimplvars,          /**< number of implicit integer variables */
   int                ncontvars,          /**< number of continuous variables */
   SCIP_VAR**         fixedvars,          /**< array with fixed and aggregated variables */
   int                nfixedvars,         /**< number of fixed and aggregated variables in the problem */
   int                startnvars,         /**< number of variables existing when problem solving started */
   SCIP_CONS**        conss,              /**< array with constraints of the problem */
   int                nconss,             /**< number of constraints in the problem */
   int                maxnconss,          /**< maximum number of constraints existing at the same time */
   int                startnconss,        /**< number of constraints existing when problem solving started */
   SCIP_Bool          genericnames,       /**< using generic variable and constraint names? */
   SCIP_RESULT*       result              /**< pointer to store the result of the file reading call */
   )
{
   *result = SCIP_DIDNOTRUN;

   return SCIP_OKAY;
}
