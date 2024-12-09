import vpi
import numpy as np

class remove_intrack():
    def remove_duplicates(self,lst):
        """
        This function removes duplicates from the list while preserving the order of the elements.
        
        :param lst: The list from which duplicates need to be removed.
        :return: A list with duplicates removed, keeping only the first occurrence.
        """
        seen = set()  # Set to track seen values
        result_out = []
        
        for item in lst:
            if item not in seen:
                result_out.append(item)  # Add item to result if not already seen
                seen.add(item)       # Mark item as seen
        return result_out

    def remove_featurecpu(self,featurecpu,row_index_list):
        featurecpu = np.delete(featurecpu, row_index_list, axis=0)
        featurecpu = vpi.asarray(featurecpu)

        return featurecpu
    def remove_status(self,sat,row_index_list):
        sat = np.delete(sat, row_index_list, axis=0)
        sat = vpi.asarray(sat)
        return sat
    def remove_id_list_intrack(self,id_list_intrack,row_index_list):
        #removed_element=id_list_intrack.pop(int(row_index))
        id_list_intrack = [row for i, row in enumerate(id_list_intrack) if i not in row_index_list]

        return id_list_intrack
    def remove_history_old_point(self,history,row_index_list):
        history= [row for i, row in enumerate(history) if i not in row_index_list]
        return history
    
    def apply_remove(self,idx_list_need_remove,featurecpu,sat,id_list_intrack,history):
        
        idx_list_need_remove=self.remove_duplicates(idx_list_need_remove)
        with featurecpu.lock_cpu() as curFeatures_cpu:
            with sat.lock_cpu() as status_cpu:
                
                    
                removed_status=self.remove_status(status_cpu,idx_list_need_remove)
                removed_featurecpu=self.remove_featurecpu(curFeatures_cpu,idx_list_need_remove)
                removed_history=self.remove_history_old_point(history,idx_list_need_remove)
                removed_id_intrack=self.remove_id_list_intrack(id_list_intrack,idx_list_need_remove)


        return removed_featurecpu,removed_status,removed_id_intrack,removed_history
    def apply_remove_first_step(self,id_list_need_remove,featurecpu,id_list_intrack,history):
        indices_to_remove = [index for index, value in enumerate(id_list_intrack) if value in id_list_need_remove]
        removed_featurecpu = [value for index, value in enumerate(featurecpu) if index not in indices_to_remove]
        removed_id_intrack = [value for index, value in enumerate(id_list_intrack) if index not in indices_to_remove]
        removed_history = [value for index, value in enumerate(history) if index not in indices_to_remove]
        


        return removed_featurecpu,removed_id_intrack,removed_history
    
    def remove_key_in_dict(self,dict_inside,list_key):
        filtered_dict = {key: value for key, value in dict_inside.items() if key not in list_key}
        return filtered_dict 