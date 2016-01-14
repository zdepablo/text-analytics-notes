# Document analysis 

Es index documents automatically and efficiently

Es supports database queries (SQL selection ) + full text queries + aggregation although a different type + pagination vs limit

In databases you define the datatypes in the physycal schema - data types and constraints => CREATE TABLE

In search engines, datatypes are defined in mappings
Besides ES, do not require schema definition, it can infer schema: dynamic mapping => Good for development
However, it is adviable to define schemas for search performance


##Mapping

Define the datatypes for a JSON document but also what kind of processing is used before indexing => analysis
Usual datatypes 

   * Exact values 
     * String: string
     * Whole number: byte, short, integer, long
     * Floating-point: float, double
     * Boolean: boolean
     * Date: date  / format 


   * Full Text
      * Index
      * Analyze

   * Complex types
      * Null values
      * Arrays
      * Objects 


### Get a mapping / eg. inferred 

### Update a mapping 
Requires indexing 

### Test a mapping


##Analysis
 Define the processing to perform in full text fields before indexing
 How to transform a text into tokens

## Basic analysis : Whitespace Tokenization 

## A more complex example 

## A more complete view 

A pipeline of processing stages
  * Character filters 
  * Tokenizer
  * Token filters 
     * Stopwords
     * Stemming
     * Add synonyms
     * Map  
  
## Built-in analyzers
 * Standard analyzer
 * Simple analyzer
 * Whitespace analyzer
 * Language analyzer 

## Indexing complex objects 

## Indexing arrays

## Indexing objects 

## Indexing objects with arrays 


##Query DSL

## Multi field queries _all

## Muti index and multi type queries

