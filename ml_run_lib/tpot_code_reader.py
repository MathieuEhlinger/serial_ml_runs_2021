# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
global exported_pipeline_hattori_hanzo_yall
def tpot_code_execution(path,verbosity=0):
    #Function that takes the output code of a tpot run and returnsthe output pipeline
#    try :
        global exported_pipeline_hattori_hanzo_yall
        with open(path,'r') as infile :
            file_string = infile.read()
        all_lines = file_string.split('\n')
        import_lines = [i for i in all_lines if i.split(' ')[0]=='import' or i.split(' ')[0]=='from']
         
        starting_line_pipeline= [num for num,i in enumerate(all_lines) if i.split(' ')[0]=='exported_pipeline'][0]
        
        if all_lines[starting_line_pipeline][-1]=='(' :
            ending_line_pipeline= [num for num,i in enumerate(all_lines) if i==')'][0]
            
            definition_lines=all_lines[starting_line_pipeline:ending_line_pipeline]
            definition_lines.insert(0,'global exported_pipeline_hattori_hanzo_yall')
            definition_lines[1]='exported_pipeline_hattori_hanzo_yall = make_pipeline('
            final_executable='\n'.join(import_lines+definition_lines)+')'
            if verbosity > 4 : print(final_executable)
            exec(final_executable)
            
            return exported_pipeline_hattori_hanzo_yall
            
        elif all_lines[starting_line_pipeline][-1]==')' :
            
            
            definition_line    =  [all_lines[starting_line_pipeline]]
            definition_line[0] =  definition_line[0].replace('exported_pipeline', 'exported_pipeline_hattori_hanzo_yall')
            definition_line.insert(0,'global exported_pipeline_hattori_hanzo_yall')
            final_executable='\n'.join(import_lines+definition_line)
            #print(final_executable)
            if verbosity > 4 : print(final_executable)
            exec(final_executable)
            
            return exported_pipeline_hattori_hanzo_yall
            
#    except :
#        print('Salvaging failed here ...')
#        return None
        
def tpot_model_score(path):
    #Function that takes the output code of a tpot run and returnsthe output pipeline
#    try :
        with open(path,'r') as infile :
            file_string = infile.read()
        all_lines = file_string.split('\n')
        temp_line = [i for i in all_lines if len(i.split(' '))>2]
        score_line = [i for i in temp_line if i.split(' ')[1]=='Average'][0]
        return score_line.split(':')[-1]

            