CREATE (:Concept {name: 'ODF Template File (ODT)', type: 'Document'});
CREATE (:Concept {name: 'ODF', type: 'Document'});
CREATE (wizard:Tool {name: 'Template Creation Wizard', type: 'Tool'});
CREATE (customisation:Feature {name: 'Include or Omit Plot Elements', type: 'Feature'});

MATCH (odt:Concept {name:'ODF Template File (ODT)'})
MATCH (odf:Concept {name:'ODF'})
CREATE (odt) - [:CREATED_FROM {description: 'An ODT can be created from an ODF using a wizard'}] -> (odf);

MATCH (odt:Concept {name:'ODF Template File (ODT)'})
MATCH (odf:Concept {name:'ODF'})
MATCH (wizard:Tool {name: 'Template Creation Wizard', type: 'Tool'})
MATCH (customisation:Feature {name: 'Include or Omit Plot Elements', type: 'Feature'})
CREATE (wizard)-[:FACILITATES {
  description: 'The wizard helps users create an ODT from an existing ODF.'
}]->(odt)
CREATE (wizard)-[:USES]->(odf)
CREATE (wizard)-[:ENABLES]->(customisation);

CREATE (:Content {name:'library information'});
CREATE (:Content {name: 'view file contents'});
CREATE (:Content {name: 'ini file settings'});

MATCH (lib_info:Content {name: 'library information'})
MATCH (view_file_contents:Content {name: 'view file contents'})
MATCH (ini_file_settings:Content {name: 'ini file settings'})
MATCH (odt:Concept {name:'ODF Template File (ODT)'})
CREATE (odt) - [:CONTAINS] -> (lib_info)
CREATE (odt) - [:CONTAINS] -> (view_file_contents)
CREATE (odt) - [:CONTAINS] -> (ini_file_settings);

CREATE (:Content {name: 'headers'});
CREATE (:Content {name: 'lithology'});
CREATE (:Concept {name: 'modifiers'});
CREATE (:Concept {name: 'structures'});
CREATE (:Concept {name: 'symbols'});

MATCH (lib_info:Content {name: 'library information'})
MATCH (header:Content {name: 'headers'})
MATCH (lithology:Content {name: 'lithology'})
MATCH (modifier:Concept {name: 'modifiers'})
MATCH (structures:Concept {name: 'structures'})
MATCH (symbols:Concept {name: 'symbols'})
CREATE (lib_info) - [:CONTAINS] -> (header)
CREATE (lib_info) - [:CONTAINS] -> (lithology)
CREATE (lib_info) - [:CONTAINS] -> (modifier)
CREATE (lib_info) - [:CONTAINS] -> (structures)
CREATE (lib_info) - [:CONTAINS] -> (symbols);

CREATE (:Content {name: 'track layout information'});
CREATE (:Content {name: 'depth and screen units'});
CREATE (:Content {name: 'scale and pen information (optional)'});

MATCH (lib_info:Content {name: 'view file contents'})
MATCH (track_layout:Content {name: 'track layout information'})
MATCH (depth_and_screen:Content {name: 'depth and screen units'})
MATCH (scale_and_pen:Content {name: 'scale and pen information (optional)'})
CREATE (lib_info) - [:CONTAINS] -> (track_layout)
CREATE (lib_info) - [:CONTAINS] -> (depth_and_screen)
CREATE (lib_info) - [:CONTAINS] -> (scale_and_pen);

CREATE (:Content {name: 'curve defaults'});
CREATE (:Content {name: 'computed curves'});
CREATE (:Content {name: 'table definitions'});

MATCH (ini_file_settings:Content {name: 'ini file settings'})
MATCH (curve_defaults:Content {name: 'curve defaults'})
MATCH (computed_curves:Content {name: 'computed curves'})
MATCH (table_definitions:Content {name: 'table definitions'})
CREATE (ini_file_settings) - [:CONTAINS] -> (curve_defaults)
CREATE (ini_file_settings) - [:CONTAINS] -> (computed_curves)
CREATE (ini_file_settings) - [:CONTAINS] -> (table_definitions);

CREATE (:ViewFile {name: "VIEW File"})-[:CONTAINS]->(:TrackLayout {type: "Track Layout"});
MATCH (odt:ODT), (view:ViewFile {name: "VIEW File"}) CREATE (odt)-[:MORE_COMPLEX_THAN]->(view);

MATCH (odt:ODT {name: "Generated Template"})
MATCH (wiz:Wizard {name: "Template Creation Wizard"})
MERGE (wiz)-[:CREATES]->(odt);

MATCH (docTree:DocumentInfoTree {name: "ODT Validator"})
MATCH (odt:ODT {name: "Generated Template"})
MERGE (docTree)-[:ANALYSES]->(odt);

MATCH (svc:Service {name: "GEONet"})
MATCH (odt:ODT {name: "Generated Template"})
MERGE (svc)-[:CUSTOMIZES]->(odt);

MATCH (warn1:Warning {message: "Missing headers"})
MATCH (odt:ODT {name: "Generated Template"})
MERGE (warn1)-[:DETECTED_IN]->(odt);

MATCH (warn2:Warning {message: "Incorrect modifier settings"})
MATCH (odt:ODT {name: "Generated Template"})
MERGE (warn2)-[:DETECTED_IN]->(odt);