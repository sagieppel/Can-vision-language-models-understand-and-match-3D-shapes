# Create 4 panel quiz in which the goal is to find which panel B,C,D  is identical to the object in panel a in term of some property
# Assume the images already given and arrange according to class/object_instatnce/image.jpg

import cv2
import os
import random
import numpy as np
#------------------------------------------------------------------------------------------------------------
def add_text(image,text):

    font = cv2.FONT_HERSHEY_PLAIN #cv2.FONT_HERSHEY_SIMPLEX
    #cv2.FONT_HERSHEY_PLAIN
    font_scale = 6  # Adjust size as needed
    font_thickness = 3
    color = (0, 0, 0)  # Black color for text

    # Position the text in the top-left corner
    x, y = 10, 80  # Offset for placement; adjust y based on font size

    # Add the text to the image
    cv2.putText(image, text, (x, y), font, font_scale, color, font_thickness)
    return image


#-------------------------------------------------------------------------------------
class make_quize():
    def __init__(self,main_dir = r"/media/deadcrow/6TB/python_project/Can_LVM_See3D/10_EVERYTHING_DIFFERENT//",neg_same_cat = False, max_img_per_object=3):
        self.main_dir=main_dir # main folder with image divided into main_dir/class_dir/object_dir/instance_image.png
        self.neg_same_cat = neg_same_cat # when creating negative example use different objects but belonging to the same class (else negative examples will be of different classes)
        self.max_img_per_object=max_img_per_object # maximum questions that will be generated for same object as anchor (maximum instances of object that will be use in panel A)
        #=========================Create dictionary of all images===============================
        self.lcats={} # structure that will contain all images arrange by class and object instance
        self.list_indx = []  # list of all indexes in lcats
        for cdr in os.listdir(main_dir):
            cat_dr = main_dir + "//" + cdr + "//"
            if not os.path.isdir(cat_dr): continue
            self.lcats[cdr] = {} # structure containing all images divided to class and object
            for odr in os.listdir(cat_dr):
                obj_dr = cat_dr + "//" + odr + "//"
                if not os.path.isdir(obj_dr): continue
                self.lcats[cdr][odr]=[]
                for ifl in os.listdir(obj_dr):
                    if ".jpg" in ifl:
                         self.lcats[cdr][odr].append(obj_dr +"/"+ifl)
                         self.list_indx.append({"cat":cdr,"obj":odr,"ins_num":len(self.lcats[cdr][odr]),"file":obj_dr +"/"+ifl})
            self.indx=0 # index of current class
            self.all_cats=list(self.lcats.keys())
            self.finish = False
##################################################################################################################################################
#==========================Go over all images and make one question per image
    def get_next_question(self):
        if self.indx>=len(self.list_indx): return False,0,0
        if self.indx == len(self.list_indx)-1: self.finish=True
        img_data = self.list_indx[self.indx]
        ct = img_data["cat"] # anchor image  category
        ob = img_data["obj"]  # anchor image object
        anc_path = img_data["file"] # anchor imager path
        self.img_file = img_data["file"]
        nn =  img_data["ins_num"] # anchor instance number
        self.indx += 1
        if len(self.lcats[ct][ob])<2: return self.get_next_question() # there need to be at least two instance of same object for question to generated
        if nn >= self.max_img_per_object:  return self.get_next_question() # limit number of question per object

        #--------------------Select anchor image and positive image-------------------
        anch_im = cv2.imread(anc_path)

        while(True):
                 pos_path = random.choice(self.lcats[ct][ob])
                 if pos_path!= anc_path:
                     pos_im = cv2.imread(pos_path)
                     break
        #---------------Select two negative images
        neg_im=[]
        while (True):
            if self.neg_same_cat:
                ct1=ct
            else:
               ct1 = random.choice(self.all_cats)
            obj1 = random.choice(list(self.lcats[ct1]))
            if ct1 == ct and obj1 == ob: continue
            if len(self.lcats[ct1][obj1]) == 0: continue
            neg_im.append(cv2.imread(random.choice(self.lcats[ct1][obj1])))
            break

        while (True):
                if self.neg_same_cat:
                    ct1=self.ct
                else:
                   ct1 = random.choice(self.all_cats)
                obj1 = random.choice(list(self.lcats[ct1]))
                if ct1 == ct and obj1 == ob: continue
                if len(self.lcats[ct1][obj1])==0: continue
                neg_im.append(cv2.imread(random.choice(self.lcats[ct1][obj1])))
                break

        #----------Create final image--------------------------------------------------------
        pos={'B':[0,512],'C':[512,0],'D':[512,512]}
        choices = ['B','C','D']
        answer=random.choice(choices)


        full_im=np.zeros([1024,1024,3],dtype=np.uint8)
        full_im[:512,:512] = add_text(anch_im,"A")
        y,x= pos[answer]
        full_im[y:y+512,x:x+512]=   add_text(pos_im, answer)

        nneg=0
        for ky in choices:
            if ky != answer:
                y, x = pos[ky]
                full_im[y:y + 512, x:x + 512] = add_text(neg_im[nneg], ky)
                nneg+=1
                if nneg>=len(neg_im):break
        return True,full_im, answer
##########################################################################################################################################################################
#######################################Run full quiz, given a function that receive image and output answer run the full test##############################################
    def  run_test(self,answer_question,ouput_dir="",save_error=False,save_all=False):
           #-------------------Output dirs for logs--------------------------------------------------------------------------
            if save_error or save_all:
               import shutil
               if os.path.exists(ouput_dir): shutil.rmtree(ouput_dir)
               os.mkdir(ouput_dir)
               error_dir = ouput_dir + "//errors//"
               if save_error and not os.path.exists(error_dir): os.mkdir(error_dir)

           #----------------------------------------------------------------------------------------------

            num_correct = 0
            num_invalid = 0
            num_false = 0
            num_questions = 0
            while (not self.finish):
                success, quiz_image, answer = self.get_next_question() # get question image
                if not success: break

                if "function" in str(answer_question):
                     ky,logs_txt = answer_question(quiz_image) # get answer from function
                else:
                     ky,logs_txt = answer_question.answer_question(quiz_image)  # get answer from class
                logs_txt = "\n" + str(num_questions+1) + "):" + self.img_file + "\nCorrect answer: " + answer + "\n" + logs_txt

                if save_all:
                    log_file = open(ouput_dir + "//" + str(num_questions+1) + ".txt","w")
                    log_file.write(logs_txt)
                    log_file.close()
                    cv2.imwrite(ouput_dir + "//" + str(num_questions+1) +"_Answer_"+answer+".jpg",quiz_image)
                if  ky.lower() == answer.lower():
                    num_correct += 1
                    logs_txt += "\nCorrect Answer\n"
                elif ky.lower() in ['b', 'c', 'd']:
                    print("false")
                    num_false += 1
                    logs_txt += "\nWrong Answer\n"
                    if save_error:
                        log_file = open(error_dir + "//" + str(num_questions+1) + ".txt","w")
                        log_file.write(logs_txt)
                        log_file.close()
                        cv2.imwrite(error_dir + "//" + str(num_questions+1) +"_Answer_" + answer+ ".jpg",quiz_image)
                else:
                    print("invalid choice")
                    num_invalid += 1
                num_questions += 1
         #       for ddd in range(10):
                print(logs_txt)
                print(num_questions, ")")
                print("correct=", num_correct, " Correct ratio=", num_correct / (num_correct + num_false+0.000001), " ALL=",
                      num_correct / num_questions)
                print("invalid=", num_invalid, "invalid ration=", num_invalid / num_questions)
            print("correct=", num_correct, " Correct ratio=", num_correct / (num_correct + num_false+0.000001), " ALL=", num_correct / num_questions)
            out_dic =  {"num_questions":num_questions,"num false":num_false,"num invalid":num_invalid, "invalid ratio":(num_invalid / num_questions),"correct" : num_correct, " Correct ratio": num_correct / (num_correct + num_false+0.00001), "ALL": num_correct / num_questions}
            if len(ouput_dir):
                 log_file=open(ouput_dir+"/logs.txt","w")
                 log_file.write(str(out_dic))
                 log_file.close()
            return {"num_questions":num_questions,"num false":num_false,"num invalid":num_invalid, "invalid ratio":(num_invalid / num_questions),"correct" : num_correct, " Correct ratio": num_correct / (num_correct + num_false), "ALL": num_correct / num_questions}





###########################################################################################################################################################
def answer_question(image):
    while (True):
        cv2.destroyAllWindows()
        cv2.imshow("Please choose the panel letter that contain object with identical 3D shape to the object in panel A", image)
        ky = cv2.waitKey()
        if chr(ky).lower() in ['b', 'c', 'd']:
            return str(chr(ky)), "Please choose the panel letter that contain object with identical 3D shape to the object in panel A:\nUser Answer: " +str(chr(ky))

#####################################################################################################################################################################
if __name__ == '__main__':
    in_dir = r"../test_images/"
    ouput_dir = "Quiz_results//"
    quiz_maker=make_quize(main_dir = in_dir,neg_same_cat = False, max_img_per_object=3)
    quiz_maker.run_test(answer_question,ouput_dir=ouput_dir,save_error=True,save_all=True)


