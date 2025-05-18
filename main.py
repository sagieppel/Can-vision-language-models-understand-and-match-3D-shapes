
# Generate images of the same 3D object/shape but with  differnt orientation materials and enviroments
# See [Do large language vision models understand 3D shapes?](https://arxiv.org/pdf/2412.10908) for more details.
# note that this script run with the blend file and refers to some prexisting nodes in the blend it the .py file cannot run stand alone and most run from the blender .blend
###############################Dependcies######################################################################################

import bpy
import math
import numpy as np
import bmesh
import os
import shutil
import random
import json
import sys
filepath = bpy.data.filepath
homedir = os.path.dirname(filepath)
sys.path.append(homedir) # So the system will be able to find local imports
os.chdir(homedir)
import MaterialsHandling as Materials
import ObjectsHandling as Objects
import RenderingAndSaving 
import SetScene
import time
########################################################################################

def ClearMaterials(KeepMaterials): # clean materials from scene
    mtlist=[]
    for nm in bpy.data.materials: 
        if nm.name not in KeepMaterials: mtlist.append(nm)
    for nm in mtlist:
        bpy.data.materials.remove(nm)

################################################################################################################################################################

#                                    Main 

###################################################################################################################################################################


#------------------------Input parameters---------------------------------------------------------------------

# Example HDRI_BackGroundFolder and PBRMaterialsFolder  and ObjectsFolder folders should be in the same folder as the script. 

HDRI_BackGroundFolder=r"HDRI_BackGround//" # Background hdri folder

ObjectMainFolder = r"objects//" #Folder of objects (like objaverseshapenet) 

pbr_folders = [r'PBR_Materials//']# folders with PBR materiall each folder will be use with equal chance


OutFolder="output_images/" # folder where generated images will be save

#****************************************************************************************************************
#HDRI_BackGroundFolder=r"/media/deadcrow/6TB/Materials_Assets/4k//"
#ObjectMainFolder=r"/media/deadcrow/6TB/Datasets/ObjaverseV1/"#/media/deadcrow/6TB/Datasets/ObjaverseV1/Categories//"
#OutFolder="/media/deadcrow/6TB/python_project/Can_LVM_See3D/All_Tests/10_Everythiong_Different_New_Big_Set_WITH_RESIZE_DISPLACEMENT/" # folder where out put will be save
#pbr_folders = [r'/media/deadcrow/SSD_480GB/PBR_Materials_SEAMLESS_larger_then_512pix//'] # folders with PBR materiall each folder will be use with equal chance

#OutFolder="output_images/" # folder where generated images will be save

#************************************************************



max_renders = 3 # Number of image renders per instance 
max_object = 6 # Max number of objects that will be rendered in each class
max_class = 40 # Max number of classes

orientation = 0 # each instance will have different random orientation
texture_indx = 9 # index for the texture file that will be loaded for all object (number of the PBR Library)
background_indx = 9# each instance will have different random background (the index is the number of the background HDRI file)

random_orient = True # each instance will have different random orientation
random_texture = True # each instance will have different random texture
random_background = True # each instance will have different random texture
replace_texture=True # replace the object original texture with single pbr
save_masks=True # save objecty mask
random_resize=True #  random resize object (or just just move camera)

use_priodical_exits = False # Exit blender once every few sets to avoid memory leaks, assuming that the script is run inside Run.sh loop that will imidiatly restart blender fresh
 

#==================================================================================================================================================================
#------------------Create PBR list-------------------------------------------------------- 
materials_lst = [] # List of all pbr materials folders path
for fff,fold in enumerate(pbr_folders): # go over all super folders 
    materials_lst.append([]) 
    for sdir in  os.listdir(fold): # go over all pbrs in folder
        pbr_path=fold+"//"+sdir+"//"
        if os.path.isdir(pbr_path):
              materials_lst[fff].append(pbr_path)
              
#------------------------------------Create list with all hdri files in the folder-------------------------------------
hdr_list=[]
for hname in os.listdir(HDRI_BackGroundFolder): 
   print("#####",hname)
   if ".hdr" in hname or ".exr" in hname:
         hdr_list.append(HDRI_BackGroundFolder+"//"+hname)
#x=wfwefwef     
#print("********************************************************************")
#print(hdr_list)
#print("********************************************************************")
#x=wewew


#################################Other parameters#########################################################
if not random_texture:
     materials_lst[0] = materials_lst[0][texture_indx:texture_indx+1]

if not random_background:     
       hdr_list = hdr_list[background_indx:background_indx+1]
     


NumSimulationsToRun=100000000000              # Number of simulation to run

#==============Set Rendering engine parameters (for image creaion)==========================================

bpy.context.scene.render.engine = 'CYCLES' # 
bpy.context.scene.cycles.device = 'GPU' # If you have GPU 

# Ensure we're using the correct context
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL' # Help create bumpiness texture but in some cases will also lead to deformation of object
bpy.context.scene.cycles.samples = 120 #200, #900 # This work well for rtx 3090 for weaker hardware this can take lots of time
bpy.context.scene.cycles.preview_samples = 900 # This work well for rtx 3090 for weaker hardware this can take lots of time

bpy.context.scene.render.resolution_x = 512 # output image size 
bpy.context.scene.render.resolution_y = 512 # output image size

#bpy.context.scene.eevee.use_ssr = True
#bpy.context.scene.eevee.use_ssr_refraction = True
#bpy.context.scene.cycles.caustics_refractive=True
#bpy.context.scene.cycles.caustics_reflective=True
bpy.context.scene.cycles.use_preview_denoising = True
bpy.context.scene.cycles.use_denoising = True


# get_devices() to let Blender detects GPU device
#bpy.context.preferences.addons["cycles"].preferences.get_devices()
#print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
#for d in bpy.context.preferences.addons["cycles"].preferences.devices:
#    d["use"] = 1 # Using all devices, include GPU and CPU
#    print(d["name"], d["use"])

#---------------List of materials that are part of the blender structure and will not be deleted------------------------------------------
MaterialsList=["PbrMaterial1","Black","White"] # Material graphs that will be used

#-------------------------Create output folder--------------------------------------------------------------


if not os.path.exists(OutFolder): os.mkdir(OutFolder)



#----------------------------------------------------------------------
######################Main loop##########################################################\
# loop 1: select materials, loop 2: create scences, loop 3: set materials ratios and render
 # Set the device_type
#bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"


scounter=0 # Count how many scene have been made
################### Category ###########################################################
for ncat,cat in enumerate(os.listdir(ObjectMainFolder)):    
    if ncat>max_class: break #  number of  class limit
    InCatFolder = ObjectMainFolder + "//" + cat +"//"
    OutCatFolder = OutFolder + "//" + cat +"//"
    if not os.path.exists(OutCatFolder): os.mkdir(OutCatFolder)
    #----------------------------Create list of Objects that will be loaded during the simulation---------------------------------------------------------------
#    ObjectList={}
#    ObjectList=Objects.CreateObjectList(InCatFolder)
#    print("object ",cat,"list len",len(ObjectList))


   
  


    

########### Object #####################################################
      

    for nobj,obj_fl in enumerate(os.listdir(InCatFolder)):
                   if nobj>max_object: break # number of object per class limits
                   OutObjectFolder = OutCatFolder + "//" + obj_fl[:-4]+"//"
                   if not os.path.exists(OutObjectFolder): os.mkdir(OutObjectFolder)
                   for n_im in range(max_renders):
                           if os.path.exists(OutObjectFolder+"//"+str(n_im)+'.jpg'): continue
                           
                           SetScene.CleanScene()  # Delete all objects in scence
                         
                           object_path = InCatFolder + "//" + obj_fl
                           print(object_path)
                           MainObjectName=Objects.LoadObject([0,0,0],1,object_path) # LOAD OBJECT
                           
                           SetScene.AddBackground(hdr_list) # Add randonm Background hdri from hdri folder
                      ###     SetScene.AddGroundPlane("Ground",x0=0,y0=0,z0=-0.5,sx=5+np.random.rand()*5,sy=5+np.random.rand()*5)
                     
                       #*****************88            
                           for i in range(3): # set object orientation
                                      if random_orient:
                                             orientation = np.random.rand()*6.2831853#/3
                                      bpy.data.objects[MainObjectName].rotation_quaternion[i] = orientation
                            # randomize background orientation
                           for i in range(3):    
                                         if random_background:                            
                                                   bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[i] = np.random.rand()*6.2831853  # randomize background orientation
                                         else:
                                             bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[i] = 0
                          # bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[1] = 3.14
                           bpy.data.worlds["World"].node_tree.nodes["Background.001"].inputs[1].default_value = 1


                           #bpy.data.objects[MainObjectName].rotation_euler=(0,0,0)
                            
                           if replace_texture or random_texture:
                               ClearMaterials(KeepMaterials=MaterialsList)
                               MainObject = bpy.data.objects[MainObjectName]     
                               Materials.ReplaceMaterial(MainObject,bpy.data.materials['PbrMaterial1']) # replace material on object
                               Materials.load_random_PBR_material(bpy.data.node_groups["Phase1"],materials_lst)
                 
                             
                      #***************************        

                #...........Set Scene and camera postion..........................................................
                           
                           bpy.context.scene.render.engine = 'CYCLES'
                           if random_resize: # not actually resize but different camera position
                                SetScene.SetCamera(name="Camera", lens = 32, location=(0,0,np.random.rand()*1+0.85),rotation=(0, 0, 0),shift_x=0,shift_y=0)
                           else:
                                SetScene.SetCamera(name="Camera", lens = 32, location=(0,0,1.2),rotation=(0, 0, 0),shift_x=0,shift_y=0)
#                      

#                           bpy.data.objects['Camera'].location=(0, 0, 20)
#                           bpy.data.objects['Camera'].rotation_euler=[0,0,0]
            
            #-------------------------------------------------------Save images--------------------------------------------------------------    
                        
                           RenderingAndSaving.RenderImageAndSave(FileNamePrefix=str(n_im),OutputFolder=OutObjectFolder) # Render image and save
                          ### x=sfdsfds
                           if save_masks:  RenderingAndSaving.SaveObjectFullMask([MainObjectName],OutObjectFolder + "/"+str(n_im) +"_MASK.png")
               
  
        #-------------Delete all objects from scene but keep materials---------------------------
                           objs = []
                           for nm in bpy.data.objects: objs.append(nm)
                           for nm in objs:  bpy.data.objects.remove(nm)
                           imlist=[]
                           for nm in bpy.data.images: imlist.append(nm) 
                           for nm in imlist:
                                    bpy.data.images.remove(nm)
                                # Clean materials

                           ClearMaterials(KeepMaterials=MaterialsList)
                           print("========================Finished==================================")
                           SetScene.CleanScene()  # Delete all objects in scence
                           #-----------------Priodic  Quit help to flash memory, and avoid system getting slower over time, if you run from run.sh
                           scounter+=1
                           if use_priodical_exits and scounter>=300: # Break program and exit blender, allow blender to remove junk
                                      print("priodic quire")
                                    #  time.sleep(30)
                                      bpy.ops.wm.quit_blender()
#if use_priodical_exits:
#       print("quit")
#       bpy.ops.wm.quit_blender()
#          #  break