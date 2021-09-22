# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:51:41 2021

@author: kmkedz
"""

import os
import time
import napari
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui

import napari_display_functions as my_napari
import general_functions as gen
import gallery_functions as g_f

# this variables will be injected from the main
global exp_dir
global df_name
global channel_list
global graph_list
global object_properties
global time_threshold

global viewer

##########################################################################
# specify for optional saving

def change_status(message):
    
    viewer.status = message

def should_i_save():
    
    global time_threshold
    
    mod_time = os.stat(os.path.join(exp_dir,'df',df_name)).st_mtime
    current_time = time.time()
    
    time_passed = current_time - mod_time
    
    if time_passed > time_threshold:
        
        return True
    else:
        return False

@thread_worker(connect={"returned": change_status})
def save_update():
    
    if should_i_save():
        
        df.to_pickle(os.path.join(exp_dir,'df',df_name))
    
        message = 'Data has been saved.'
    
    else:
        
        message = None
        
    return message

##########################################################################
# functions for buttons

def save_data(viewer: Viewer):
    
    global df
    global exp_dir
    global df_name
    
    df.to_pickle(os.path.join(exp_dir,'df',df_name), protocol=4)
    viewer.status = 'Data has been saved.'
       
def cut_track(viewer: Viewer):
    
    global df
    global gen_track_columns
    
    viewer,df = my_napari.cut_track(viewer,df,gen_track_columns)
    
    # optionally save dataframe
    if should_i_save():
        save_update()

def merge_track(viewer: Viewer):
    
    global df
    global gen_track_columns
    
    viewer,df = my_napari.merge_track(viewer,df,gen_track_columns)
    
    # optionally save dataframe
    if should_i_save():
        save_update()
    
def connect_track(viewer: Viewer):
    
    global df
    global gen_track_columns
    
    viewer,df = my_napari.connect_track(viewer,df,gen_track_columns)
    
    # optionally save dataframe
    if should_i_save():
        save_update()

def mod_label(viewer: Viewer):
    
    global df
    global channel_list
    global object_properties
    global gen_track_columns

    viewer,df = my_napari.update_single_object(viewer,df,channel_list,object_properties,gen_track_columns)
    
    active_label = viewer.layers['Labels'].selected_label
    viewer.status = f'Label {active_label} has been modified.'
    
    # optionally save dataframe
    if should_i_save():
        save_update()
    
def select_label(layer, event):
    
    global viewer

    if(event.button == 2):
        
        # get data
        inProgressTracks = viewer.layers['Labels'].data
        
        # look up cursor position
        x = int(viewer.cursor.position[1])
        y = int(viewer.cursor.position[2])
        
        # check which cell was clicked
        myTrackNum = inProgressTracks[viewer.dims.current_step[0],x,y]
        
        # set track as active
        viewer.layers['Labels'].selected_label = myTrackNum
        
def toggle_track(layer, event):

    global df
    global viewer
    global tag_list
    global gen_track_columns
    
    if(event.button == 2):
    
        # look up cursor position
        t = viewer.dims.current_step[0]
        x = int(viewer.cursor.position[1])
        y = int(viewer.cursor.position[2])

        # get data
        labels_layer = viewer.layers['Labels']
        inProgressTracks = labels_layer.data
    
        # check which cell was clicked
        myTrackNum = inProgressTracks[t,x,y]
    
        if myTrackNum > 0:

            track_ind = [x['tag_name'] for x in tag_list].index(layer.name)
            tag_column = tag_list[track_ind]['tag_column']
    
            # check status
            track_status_old = list(df.loc[df.track_id==myTrackNum,tag_column])[0]
            track_status_new = not(track_status_old)
    
            # change status of this track
            df.loc[df.track_id==myTrackNum,tag_column] = track_status_new
    
            # in case a status is activated, others need to be deactivated
            if track_status_new == True:
    
                for tag in tag_list:
    
                    if (tag['tag_name'] != layer.name):
                        df.loc[df.track_id==myTrackNum,tag['tag_column']] = False
    
            ########################################################
            # modify points layer
            ########################################################
    
            # regenerate tags data
            tag_data = gen.tags_from_df(df,tag_list)
    
            # update viewer   
            for tag,tag_data in zip(tag_list,tag_data):
    
                viewer.layers[tag['tag_name']].data = tag_data
    
            ########################################################
            # modify tracking layer
            ########################################################
    
            # modify the data for the layer
            data,properties,graph = gen.trackData_from_df(df,col_list = gen_track_columns)
    
            # change tracks layer
            viewer.layers['Tracking'].data = data
            viewer.layers['Tracking'].properties = properties
            viewer.layers['Tracking'].graph = graph      
            
            # optionally save dataframe
            if should_i_save():
                save_update()

def update_stack(viewer_stack,active_track):
    
    global viewer
    global channel_list
    global small_im_size
    global label_contour
    global df
    
    labels = viewer.layers['Labels'].data
        
    # ask for an update
    stack_im_list,stack_labels = g_f.stack_create_all(channel_list,labels,df,active_track,imSize=small_im_size)
    
    # display new layers
    my_napari.display_set(viewer_stack,stack_labels,stack_im_list,channel_list,label_contour = label_contour)
      
def update_graph(viewer_stack,active_track):
    
    global df
    global graph_list
    global mpl_widget
    
    # remove previous graph
    h = viewer_stack.window._dock_widgets['']
    viewer_stack.window.remove_dock_widget(h)
        
    # add new graph
    mpl_widget = my_napari.create_graph_widget(graph_list,df,active_track) 
    h = viewer_stack.window.add_dock_widget(mpl_widget)
   
def update_stack_button_f(viewer_stack: Viewer):
    
    global viewer
    global channel_list
    global graph_list
    global small_im_size
    global label_contour
    global df
    global mpl_widget
    global position_line_list
    global graph_offset

    active_label = viewer_stack.layers['Labels'].selected_label

    # update stack
    update_stack(viewer_stack,active_label)

    # update graph
    update_graph(viewer_stack,active_label) 
    
    # calculate offset on x axis
    graph_offset = gen.calculate_graph_offset(df,active_label)
    
    # initiate the position line and connect to the function
    init_lines(viewer_stack)

def init_lines(viewer_stack):

    global mpl_widget
    global position_line_list
    global graph_offset
    
    x = viewer_stack.dims.current_step[0] + graph_offset  
 
    position_line_list = []
    for lin_num in range(len(mpl_widget.figure.axes)):
        
        position_line = mpl_widget.figure.axes[lin_num].axvline(x=x,color='black')
        position_line_list.append(position_line)
    
        # connect the function to the dims axis
        viewer_stack.dims.events.current_step.connect(update_lines)
    
def update_lines(step_event):

    global mpl_widget
    global position_line_list
    global graph_offset

    steps = step_event.value
    slice_num = steps[0] + graph_offset
    
    for position_line in position_line_list:

        current_pos = position_line.get_data()[0][0]

        if slice_num == current_pos:
            return
        position_line.set_data([slice_num, slice_num], [0, 1])

        mpl_widget.draw_idle()    

def show_stack(viewer: Viewer):
    
    global mpl_widget
    global position_line_list
    global graph_offset
    
    # find current track
    active_label = viewer.layers['Labels'].selected_label
    
    # init stack viewer
    viewer_stack = napari.Viewer()
    update_stack(viewer_stack, active_label)
    
    # set the right label
    viewer_stack.layers['Labels'].selected_label = active_label
    
    # init graph
    mpl_widget = my_napari.create_graph_widget(graph_list,df,active_label) 
    viewer_stack.window.add_dock_widget(mpl_widget)
    
    # calculate offset on x axis
    graph_offset = gen.calculate_graph_offset(df,active_label)
    
    # initiate the position line and connect to the function
    init_lines(viewer_stack)
    
    # create an update button and connect to the function
    update_stack_button = magicgui(update_stack_button_f, call_button='Update Stack')
    viewer_stack.window.add_dock_widget(update_stack_button,area='bottom')
    