CREATE (:Concept {name: 'ODF Template File (ODT)', type: 'Document'});
CREATE (:Concept {name: 'ODF', type: 'Document'});

MATCH (odt:Concept {name:'ODF Template File (ODT)'})
MATCH (odf:Concept {name:'ODF'})
CREATE (odt) - [:CREATED_FROM] -> (odf);

CREATE (:Content {name:'library information'});
CREATE (:Content {name: 'view file contents'});
CREATE (:Content {name: 'ini file settings'});

MATCH (lib_info:Content {name: 'library information'})
MATCH (view_file_contents:Content {name: 'view file contents'})
MATCH (ini_file_settings:Content {name: 'ini file settings'})
MATCH (odt:Concept {name:'ODF Template File (ODT)'})
CREATE (odt) - [:CONTAINS] -> (lib_info);
CREATE (odt) - [:CONTAINS] -> (view_file_contents);
CREATE (odt) - [:CONTAINS] -> (ini_file_settings);

CREATE (:Content {name: 'headers'})
CREATE (:Content {name: 'lithology'})
CREATE (:Concept {name: 'modifiers'})
CREATE (:Concept {name: 'structures'})
CREATE (:Concept {name: 'symbols'})

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
CREATE (lib_info) - [:CONTAINS] -> (symbols)