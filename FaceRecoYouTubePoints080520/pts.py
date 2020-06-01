# def detect_face_rect_of_pts(img_path,pts):
#     detector = dlib.get_frontal_face_detector()
#    # detect face using dlib
#     im = cv2.imread(img_path);
#     dets = detector(im, 1)
#     selected_det = None
#     pts_center = np.mean(pts,axis=0)
#     for det in dets:
#         if pts_center[0]>det.left() and pts_center[0]<det.right() and\
#         pts_center[1]>det.top() and pts_center[1]<det.bottom():
#             selected_det = det
#     if selected_det is None:
#         return None
#     left = selected_det.left()
#     top = selected_det.top()
#     width = selected_det.width()
#     height = selected_det.height()
#     return [left,top,width,height] 

# detect_face_rect_of_pts("C:\\Users\\Stas\\Univercity_cameras\\FaceRecoYouTubePoints080520\\Ellie_sattler.png")

a = "[(261, 244) (410, 394)]"
a = str(a).replace(") (",", ")
print(a)