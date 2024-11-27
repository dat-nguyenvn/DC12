import numpy as np
from wildtracker.utils.update import update_bounding_box

def reconstruct_process(np_image,dict_coco_annotations,cur_feature,tracking_list):
    #dict_inside=dict_coco_annotations
    unique_values = set(tracking_list)
    print("unique_values in reconstruct ",unique_values)
    for unique in unique_values:
        indices_of_unique = np.where(np.array(tracking_list) == unique)[0].tolist()
        
        #point_of_one_id_step_t=[tuple(prev_feature[i]) for i in indices_of_unique]
        
        
        ####
        # print("cur_feature", cur_feature.shape)
        # print("tracking_list",len(tracking_list))
        # print("set track ing list ",set(tracking_list))
        
        
        
        point_of_one_id_step_t1=[tuple(cur_feature[i]) for i in indices_of_unique]

        #dict_coco_annotations[unique-1]['points_step_t']=point_of_one_id_step_t1
        '''
        #Todo truong hop id bi xoa(mat track) co id co step t nhung khong co t+1
        for item in dict_coco_annotations:
            # Check if the image_id matches
            if item['image_id'] == unique:
                box_ori=item['ori_bbox']
                #box_previous=find_box_by_id(list_boxes_complete_step_t,unique)
                center_t0=item['ori_center_points']
                reconstructed_box_step_t1=update_bounding_box(center_t0,point_of_one_id_step_t1,box_ori)
                #x,y,w,h=reconstructed_box_step_t1
                
                simple_box_object=reconstructed_box_step_t1
                #print("reconstructed_box_step_t1",type(reconstructed_box_step_t1[0]))
                #print("reconstructed_box_step_t1",reconstructed_box_step_t1)
                item['bbox']=simple_box_object
                #dict_coco_annotations=update_bbox_by_image_id(dict_coco_annotations,unique,simple_box_object)
                # listtest=point_of_one_id_step_t1 #ok
                # listtest.append(center_t0)
                # print("listtest",listtest)
                # show_image=visual_image().visualize_points_on_image(image_np=np_image,points=listtest,color=(123,0,10))
                # plt.imshow(show_image)
                # plt.show()
            '''

        if unique in dict_coco_annotations:
            item = dict_coco_annotations[unique]
            box_ori=item['ori_bbox']
            #box_previous=find_box_by_id(list_boxes_complete_step_t,unique)
            center_t0=item['ori_center_points']
            reconstructed_box_step_t1, drift =update_bounding_box(center_t0,point_of_one_id_step_t1,box_ori)
            #x,y,w,h=reconstructed_box_step_t1
            item['drift']=drift
            simple_box_object=reconstructed_box_step_t1
            #print("reconstructed_box_step_t1",type(reconstructed_box_step_t1[0]))
            #print("reconstructed_box_step_t1",reconstructed_box_step_t1)
            item['bbox']=simple_box_object
    return dict_coco_annotations