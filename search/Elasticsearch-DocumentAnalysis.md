# Document analysis 

Es index documents automatically and efficiently

Es supports database queries (SQL selection ) + full text queries + aggregation although a different type + pagination vs limit

In databases you define the datatypes in the physycal schema - data types and constraints => CREATE TABLE

In search engines, datatypes are defined in mappings
Besides ES, do not require schema definition, it can infer schema => Good for development
However, it is adviable to define schemas for search performance


##Mapping

Define the datatypes for a JSON document but also what kind of processing is used before indexing => analysis
Usual datatypes 

   * Exact values 
     * String
     * Long
     * Date
     * ...

   * Full Text


### Get a mapping / eg. inferred 


##Analysis

##Query DSL

## Multi field queries _all

## Muti index and multi type queries


