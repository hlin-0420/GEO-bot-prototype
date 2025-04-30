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
