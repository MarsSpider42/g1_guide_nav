import time
import cv2
import numpy as np
import open3d as o3d
# import matplotlib.pyplot as plt

class LidarProcessor:
    # Cropping Thresholds #
    MAX_DISTANCE_CROP_THRESHOLD = 6 #m
    MIN_DISTANCE_CROP_THRESHOLD = 0.3 #m

    # Standard Cloud Reorient #
    STD_TILT = -2.3 #degrees

    # Dynamic Cloud Reorient #
    GROUND_PLANE_THRESHOLD = 0.05 #m
    RANSAC_N = 5
    ITERATIONS = 100             # RANSAC iterations per plane (only used when the ground has to be searched for)

    RANSAC_VOXEL = 0.05          # m; plane SEARCH runs on a thinned cloud (0 = use every point). Inliers are still counted on the full cloud.
    # Ground tracking: reuse last frame's plane when it still fits, and only run RANSAC when it does not #
    TRACK_GROUND = True
    TRACK_MAX_TILT_DEG = 4.0       # the tracked normal may move this much in one frame
    TRACK_MAX_OFFSET_M = 0.12      # ... and the plane may move this far along its normal
    TRACK_MIN_INLIER_FRACTION = 0.85  # inliers must be at least this fraction of the floor size seen at the last search
    TRACK_BANDS = (0.25, 0.12, 0.06, 0.05)   # coarse-to-fine refit bands (m): lets the plane follow a few degrees of tilt change
    TRACK_RESEARCH_EVERY = 20      # force a full RANSAC search every N frames so a bad plane cannot stay locked in
    MAX_PLANES_TO_FIT = 8
    MIN_ALIGNMENT = np.cos(np.deg2rad(50)) #degrees

    FLAT_PLANES_REWARD_COEF = 5.0
    LARGE_PLANES_REWARD_COEF = 0.001
    LOW_CENTROID_REWARD_COEF = 0.5

    # Section Identification Thresholds #
    CEIL_THRESH = 0.68 #m

    # Height and Occupancy Grid Construction #
    G_RESOLUTION = 0.05 #m

    MIN_OBST_HEIGHT = 0.3 #m
    MAX_OBST_HEIGHT = 2.0 #m

    def __init__(self, cloud, window_duration=15, occupancy_x_extent : tuple =(0,6), occupancy_y_extent : tuple =(-3,3)):
        self.cloud = cloud # Raw LiDAR Cloud being processed
        self.show_duration = window_duration # sets length of debug shows

        self._ground_track = None   # (unit normal, offset d, inlier count) of the last accepted ground plane
        self.ground_mode = None     # "track" or "search", for the viewer / logs
        self._frames_since_search = 0

        self.G_X_MIN = occupancy_x_extent[0] #m
        self.G_X_MAX = occupancy_x_extent[1] #m
        self.G_Y_MIN = occupancy_y_extent[0] #m
        self.G_Y_MAX = occupancy_y_extent[1] #m

    "Runs LiDAR processing pipeline on stored cloud data."     
    def process(self, debug):
        LidarProcessor.__crop_cloud(self, debug)
        LidarProcessor.__reorient_standard(self, debug)
        LidarProcessor.__reorient_wrt_ground_plane(self, debug)
        LidarProcessor.__identify_sections(self, debug)
        LidarProcessor.__height_occ_grids(self, debug)

    def __clear_data(self):
        attrs_to_clear = [
            "cloud", "cropped_cloud", "std_reoriented_cloud",
            "reoriented_cloud", "ground_plane_cloud", "ceiling_cloud",
            "behind_cloud", "obstacle_cloud", "height_grid", "occupancy_grid"
        ]
        for attr in attrs_to_clear:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def update_cloud(self, new_cloud):
        self.__clear_data()
        self.cloud = new_cloud

    def show_raw(self, duration):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.cloud)
    
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5
        )
    
        print("\nShowing whole cloud...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Raw LIDAR Cloud", width=1280, height=720)
        vis.add_geometry(pcd)
        vis.add_geometry(frame)
    
        ctr = vis.get_view_control()
    
        # Smaller = further away
        # Larger = closer in
        ctr.set_zoom(0.3)
    
        # Optional camera orientation
        ctr.set_front([0, 0, -1])   # viewing direction
        ctr.set_lookat([0, 0, 0])   # centre point
        ctr.set_up([0, -1, 0])      # up direction
    
        start_time = time.time()
    
        while True:
            vis.poll_events()
            vis.update_renderer()
            if time.time() - start_time > duration:
                break
    
        print("\nClosing window...")
        vis.destroy_window()
   
    def show_cloud(self, cloud_type : str, duration=-1):
        '''Displays a given point cloud using Open3D for a specified duration.
                    cloud_type: string containing cloud type to be shown, options are\n 
                            - "cropped cloud"\n
                            - "standard reoriented cloud"\n
                            - "dynamically reoriented cloud"\n
                            - "obstacle cloud"\n
                    duration (optional): duration to show window for, if not set, will \n
                        use default set show_duration from object initialisation. '''
        
        if(duration == -1):
            duration = self.show_duration

        if(cloud_type == "cropped cloud"):
            c = self.cropped_cloud
            title = f"LiDAR Cloud Cropped to exclude points < {self.MIN_DISTANCE_CROP_THRESHOLD}m or > {self.MAX_DISTANCE_CROP_THRESHOLD}m from G1"
        elif(cloud_type == "standard reoriented cloud"):
            c = self.std_reoriented_cloud
            title = f"LiDAR Cloud with {self.STD_TILT} degree rotation applied about Y axis"
        elif(cloud_type == "dynamically reoriented cloud"):
            c = self.reoriented_cloud
            title = "LiDAR Cloud with dynamic reorientation to account for head tilt wrt ground"
        elif(cloud_type == "obstacle cloud"):
            c = self.obstacle_cloud
            title = "LiDAR Cloud of potential obstacles"
        else:
            print("Please include a valid cloud type:\n- ""cropped cloud""\n- ""standard reoriented cloud""\n- ""dynamically reoriented cloud""\n- ""obstacle cloud""")
            return


        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(c)
        vis.add_geometry(frame)

        ctr = vis.get_view_control()
        # Smaller = further away
        # Larger = closer in
        ctr.set_zoom(0.3)

        # Optional camera orientation
        ctr.set_front([0.4, -0.4, -0.6])   # viewing direction
        ctr.set_lookat([0, 0, 0])   # centre point
        ctr.set_up([0, 0, -1])      # up direction

        start_time = time.time()

        while True:
            vis.poll_events()
            vis.update_renderer()
            if time.time() - start_time > self.show_duration:
                break

        vis.destroy_window()

    def __crop_cloud(self, debug):
        ## CROP CLOUD TO REMOVE FAR POINTS ##
        distance_values = np.linalg.norm(self.cloud, axis=1)
        distance_mask = (distance_values < self.MAX_DISTANCE_CROP_THRESHOLD) & (distance_values > self.MIN_DISTANCE_CROP_THRESHOLD)
        c_cloud = self.cloud[distance_mask]    
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(c_cloud)
        self.cropped_cloud = cropped_pcd

        if debug:
            print("Points before distance crop:", len(self.cloud),"\nPoints after distance crop:", len(c_cloud))
        

    def __reorient_standard(self, debug):
        ## CREATE POINT CLOUD ##
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(self.cropped_cloud.points)

        ## CONSTANT 2.3 DEG ROTATION ABT Y ##
        R = o3d.geometry.get_rotation_matrix_from_xyz( (0, np.deg2rad(self.STD_TILT), 0))
        
        if debug:
            print(f"Std Rotation Applied: {R}")

        ## ROTATE CLOUD ##
        aligned_pcd.rotate(R, center=(0,0,0))

        self.std_reoriented_cloud = aligned_pcd

    def __plane_inliers(self, pts, plane_model):
        """Indices of ALL points within GROUND_PLANE_THRESHOLD of the plane [a, b, c, d].
        Plain element-wise arithmetic (no matrix multiply), so no BLAS threads get involved."""
        n = np.asarray(plane_model[:3], dtype=float)
        dist = np.abs(pts[:, 0] * n[0] + pts[:, 1] * n[1] + pts[:, 2] * n[2] + plane_model[3]) / np.linalg.norm(n)
        return np.flatnonzero(dist < self.GROUND_PLANE_THRESHOLD)

    def __refine_ground_plane(self, pts, plane_model, up, passes=2, bands=None):
        """Least-squares refit of the chosen plane to its own inliers; returns (unit normal pointing up, offset d, inliers).
        The fit is the smallest-eigenvalue direction of the 3x3 covariance of the inliers."""
        n = np.asarray(plane_model[:3], dtype=float)
        scale = np.linalg.norm(n)
        n, d = n / scale, plane_model[3] / scale
        widths = bands if bands is not None else [self.GROUND_PLANE_THRESHOLD] * passes
        for w in widths:
            sel = pts[np.abs(pts[:, 0] * n[0] + pts[:, 1] * n[1] + pts[:, 2] * n[2] + d) < w]
            if len(sel) < 50:
                break
            c = sel.mean(axis=0)
            r = sel - c
            cov = np.einsum("ni,nj->ij", r, r)               # 3x3, no BLAS
            n = np.linalg.eigh(cov)[1][:, 0]                 # eigenvector of the smallest eigenvalue
            d = -float(np.sum(n * c))
        if np.dot(n, up) < 0:
            n, d = -n, -d
        inliers = np.flatnonzero(np.abs(pts[:, 0] * n[0] + pts[:, 1] * n[1] + pts[:, 2] * n[2] + d) < self.GROUND_PLANE_THRESHOLD)
        return n, d, inliers

    def __reorient_wrt_ground_plane(self, debug):
        all_pts = np.asarray(self.std_reoriented_cloud.points)    # the FULL cloud, never shrinks
        estimated_up = np.array([0, 0, 1])
        best_normal = best_inliers = None
        best_i = None
        candidates = []
        t_ransac = t_recount = 0.0
        n_search = 0
        ref_count = None

        ## TRY TO TRACK LAST FRAME'S GROUND PLANE (cheap) ##
        self.ground_mode = "search"
        if (self.TRACK_GROUND and self._ground_track is not None
                and self._frames_since_search < self.TRACK_RESEARCH_EVERY):
            n0, d0, count0 = self._ground_track
            n1, d1, inl1 = self.__refine_ground_plane(all_pts, np.r_[n0, d0], estimated_up, bands=self.TRACK_BANDS)
            tilt_move = np.degrees(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0)))
            if (len(inl1) >= self.TRACK_MIN_INLIER_FRACTION * count0
                    and tilt_move <= self.TRACK_MAX_TILT_DEG
                    and abs(d1 - d0) <= self.TRACK_MAX_OFFSET_M):
                best_normal, best_d, best_inliers = n1, d1, inl1
                best_i = -1                                  # -1 = tracked, no search needed
                self.ground_mode = "track"
                ref_count = max(0.98 * count0, len(inl1))     # reference decays slowly, so a shrinking floor cannot ratchet the test down
                self._frames_since_search += 1
            else:
                candidates.append(f"TRACK FAILED inliers {len(inl1)}/{count0} tilt_move {tilt_move:.1f} offset_move {abs(d1 - d0):.2f}")

        if best_normal is None:
            ## CREATE PCD ##
            fitting_pcd = o3d.geometry.PointCloud(self.std_reoriented_cloud.points)
            if self.RANSAC_VOXEL > 0:
                fitting_pcd = fitting_pcd.voxel_down_sample(self.RANSAC_VOXEL)    # search on fewer points

            best_plane = None
            best_score = -np.inf
            n_search = len(fitting_pcd.points)          # points RANSAC starts from


            ## SEARCH MULTIPLE PLANES ##
            for i in range(self.MAX_PLANES_TO_FIT):
                if(len(fitting_pcd.points) < 500):
                    break
            
                _t = time.perf_counter()
                plane_model, inliers = fitting_pcd.segment_plane(
                    distance_threshold=self.GROUND_PLANE_THRESHOLD,
                    ransac_n=self.RANSAC_N,
                    num_iterations=self.ITERATIONS
                )
                t_ransac += time.perf_counter() - _t

                # RANSAC only sees the points earlier planes left behind, so a floor that earlier (wrong) planes
                # nibbled at gets split across several planes. Re-count this plane's inliers over the FULL cloud.
                _t = time.perf_counter()
                full_inliers = self.__plane_inliers(all_pts, plane_model)
                t_recount += time.perf_counter() - _t

                [a, b, c, d] = plane_model

                normal = np.array([a, b, c])
                normal /= np.linalg.norm(normal)

                # flip towards estimated up if required
                if np.dot(normal, estimated_up) < 0:
                    normal = -normal

                ## CALCULATE ALIGNMENT SCORE OF FITTED PLANE ##
                alignment  = np.dot(normal, estimated_up)

                if debug:
                    print(f"Plane {i}\nNormal: {normal}, Alignment: {alignment}, Num Inliers: {len(inliers)}")

                ## REJECT PROBABLE WALLS ##
                if alignment < self.MIN_ALIGNMENT:
                    candidates.append(f"#{i} REJECTED(wall) align={alignment:.2f} n={len(inliers)}")
                    fitting_pcd = fitting_pcd.select_by_index(inliers, invert = True)

                    if debug:
                        print(f"\nRejected Plane {i}, too vertical\n")
                
                    continue

                ## FIT ANOTHER HEURISTIC, REWARDING LOW PLANES, LARGE PLANES, AND FLAT PLANES ##
                plane_cloud = fitting_pcd.select_by_index(inliers)

                plane_points = np.asarray(plane_cloud.points)


                centroid = plane_points.mean(axis=0)

                score = (self.FLAT_PLANES_REWARD_COEF * alignment
                    + self.LARGE_PLANES_REWARD_COEF * len(full_inliers)
                    + self.LOW_CENTROID_REWARD_COEF * centroid[2])


                candidates.append(f"#{i} align={alignment:.2f} n={len(full_inliers)} (ransac {len(inliers)}) z={centroid[2]:.2f} score={score:.2f}")

                if debug:
                    print(f"Centroid: {centroid}, Score: {score}")

                if score > best_score:
                    best_score = score
                    best_plane = plane_model
                    best_normal = normal
                    best_inliers = full_inliers
                    best_i = i

                # Remove plane and continue searching
                fitting_pcd = fitting_pcd.select_by_index(
                    inliers,
                    invert=True
                )

            ## CHECK IF A VALID GROUND HAS BEEN FOUND ##
            if best_plane is None:
            
                raise RuntimeError("No valid ground plane found")

            ## REFINE THE WINNER: least-squares fit to its own inliers, then re-count over the full cloud ##
            _t = time.perf_counter()
            best_normal, best_d, best_inliers = self.__refine_ground_plane(all_pts, best_plane, estimated_up)
            t_refine = (time.perf_counter() - _t) * 1000

        if best_i == -1:
            t_refine = 0.0
        else:
            self._frames_since_search = 0
        self._ground_track = (best_normal, best_d, ref_count if ref_count is not None else len(best_inliers))
        self.ground_timing = {"ransac_ms": t_ransac * 1000, "recount_ms": t_recount * 1000, "refine_ms": t_refine,
                              "planes": 0 if best_i == -1 else best_i + 1, "search_points": n_search,
                              "full_points": len(all_pts), "mode": self.ground_mode}

        prev_tilt = getattr(self, "ground_tilt_deg", None)     # tilt chosen on the previous frame
        self.ground_tilt_deg = float(np.degrees(np.arccos(np.clip(best_normal[2], -1.0, 1.0))))

        ## LOG ODD FRAMES: the chosen ground tilt jumped by more than 2 degrees since the previous frame ##
        if prev_tilt is not None and abs(self.ground_tilt_deg - prev_tilt) > 2.0:
            prev_txt = "-" if prev_tilt is None else f"{prev_tilt:.1f}"
            print(f"[ground] ODD: mode {self.ground_mode}, chose #{best_i}, tilt {self.ground_tilt_deg:.1f} deg (prev {prev_txt}) | " + " | ".join(candidates))

        if debug:
            print(f"\nSelected ground plane normal from Plane {best_i}: {best_normal}\n")

        ## COMPUTE ROTATION ##
        rotation_axis = np.cross(best_normal, estimated_up)
        rotation_axis_norm = np.linalg.norm(rotation_axis) 

        if rotation_axis_norm > 1e-6: # if rotation is too small, do not apply
            rotation_axis /= rotation_axis_norm

            angle = np.arccos(np.clip(np.dot(best_normal, estimated_up), -1.0, 1.0))

            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        else:
            R = np.eye(3)

        ## APPLY ROTATION ##
        aligned_pcd = o3d.geometry.PointCloud(self.std_reoriented_cloud.points)
        aligned_pcd.rotate(R, center=(0,0,0))

        ## SEPARATE CLOUDS FOR RETURN ##
        self.ground_plane_cloud = aligned_pcd.select_by_index(best_inliers)

        self.non_ground = aligned_pcd.select_by_index(best_inliers, invert=True)
        self.best_normal = best_normal

        self.reoriented_cloud = aligned_pcd.select_by_index(
            best_inliers,
            invert=True
        )

    def __legacy_reorient_RANSAC_only(self, debug):
        # Dynamic Cloud Reorient #
        _GROUND_PLANE_THRESHOLD = 0.05 #m
        _RANSAC_N = 5
        _ITERATIONS = 300

        _MAX_PLANES_TO_FIT = 8
        _MIN_ALIGNMENT = np.cos(np.deg2rad(50)) #degrees

        _FLAT_PLANES_REWARD_COEF = 5.0
        _LARGE_PLANES_REWARD_COEF = 0.001
        _LOW_CENTROID_REWARD_COEF = 0.5

        ## CREATE PCD ##
        fitting_pcd = o3d.geometry.PointCloud(self.std_reoriented_cloud.points)

        # START VARIABLES FOR SEARCH #
        best_plane = None
        best_normal = None
        best_inliers = None
        best_score = -np.inf

        ## SET ASSUMED WORLD FRAME UP ##
        estimated_up  = np.array([0,0,1]) # currently just a vector along Z

        ## SEARCH MULTIPLE PLANES ##
        for i in range(_MAX_PLANES_TO_FIT):
            if(len(fitting_pcd.points) < 500):
                break
            
            plane_model, inliers = fitting_pcd.segment_plane(
                distance_threshold=_GROUND_PLANE_THRESHOLD,
                ransac_n=_RANSAC_N,
                num_iterations=_ITERATIONS
            )

            [a, b, c, d] = plane_model

            normal = np.array([a, b, c])
            normal /= np.linalg.norm(normal)

            # flip towards estimated up if required
            if np.dot(normal, estimated_up) < 0:
                normal = -normal

            ## CALCULATE ALIGNMENT SCORE OF FITTED PLANE ##
            alignment  = np.dot(normal, estimated_up)

            if debug:
                print(f"Plane {i}\nNormal: {normal}, Alignment: {alignment}, Num Inliers: {len(inliers)}")

            ## REJECT PROBABLE WALLS ##
            if alignment < _MIN_ALIGNMENT:
                fitting_pcd = fitting_pcd.select_by_index(inliers, invert = True)

                if debug:
                    print(f"\nRejected Plane {i}, too vertical\n")
                
                continue

            ## FIT ANOTHER HEURISTIC, REWARDING LOW PLANES, LARGE PLANES, AND FLAT PLANES ##
            plane_cloud = fitting_pcd.select_by_index(inliers)

            plane_points = np.asarray(plane_cloud.points)


            centroid = plane_points.mean(axis=0)

            score = (_FLAT_PLANES_REWARD_COEF * alignment
                + _LARGE_PLANES_REWARD_COEF * len(inliers)
                + _LOW_CENTROID_REWARD_COEF * centroid[2])


            if debug:
                print(f"Centroid: {centroid}, Score: {score}")

            if score > best_score:
                best_score = score
                best_plane = plane_model
                best_normal = normal
                best_inliers = inliers

            # Remove plane and continue searching
            fitting_pcd = fitting_pcd.select_by_index(
                inliers,
                invert=True
            )

        ## CHECK IF A VALID GROUND HAS BEEN FOUND ##
        if best_plane is None:
            
            raise RuntimeError("No valid ground plane found")

        self.ground_tilt_deg = float(np.degrees(np.arccos(np.clip(best_normal[2], -1.0, 1.0))))

        if debug:
            print(f"\nSelected ground plane normal from Plane {i}: {best_normal}\n")

        ## COMPUTE ROTATION ##
        rotation_axis = np.cross(best_normal, estimated_up)
        rotation_axis_norm = np.linalg.norm(rotation_axis) 

        if rotation_axis_norm > 1e-6: # if rotation is too small, do not apply
            rotation_axis /= rotation_axis_norm

            angle = np.arccos(np.clip(np.dot(best_normal, estimated_up), -1.0, 1.0))

            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        else:
            R = np.eye(3)

        ## APPLY ROTATION ##
        aligned_pcd = o3d.geometry.PointCloud(self.std_reoriented_cloud.points)
        aligned_pcd.rotate(R, center=(0,0,0))

        ## SEPARATE CLOUDS FOR RETURN ##
        self.ground_plane_cloud = aligned_pcd.select_by_index(best_inliers)
        
        self.non_ground = aligned_pcd.select_by_index(best_inliers, invert=True)
        self.best_normal = best_normal

        

    def __identify_sections(self, debug):

        points = np.asarray(self.reoriented_cloud.points)

        if debug:
            print("Reoriented Cloud Axis Ranges:")
            print("X:", points[:, 0].min(), "to", points[:, 0].max(), " range: ", points[:, 0].max()-points[:, 0].min(), "m")
            print("Y:", points[:, 1].min(), "to", points[:, 1].max(), " range: ", points[:, 1].max()-points[:, 1].min(), "m")
            print("Z:", points[:, 2].min(), "to", points[:, 2].max(), " range: ", points[:, 2].max()-points[:, 2].min(), "m")   
        

        ## MASK OUT POINTS BY Z COORDINATE ##
        ceil_mask = (
            points[:,2] < -self.CEIL_THRESH # if point is higher than 2m total height
        )
        behind_mask = points[:,0] < 0 # if point is in negative half of x plane

        ## FILTER POINTS BY MASKS ##
        ceil_points = points[ceil_mask]
        behind_points = points[behind_mask & (~ceil_mask)]
        other_points = points[(~ceil_mask) & (~behind_mask)]

        ## MAKE CLOUDS ##
        ceil_cloud = o3d.geometry.PointCloud()
        ceil_cloud.points = o3d.utility.Vector3dVector(ceil_points)
        floor_cloud = o3d.geometry.PointCloud(self.ground_plane_cloud.points)
        behind_cloud = o3d.geometry.PointCloud()
        behind_cloud.points = o3d.utility.Vector3dVector(behind_points)

        other_cloud = o3d.geometry.PointCloud()
        other_cloud.points = o3d.utility.Vector3dVector(other_points)

        self.ceiling_cloud = ceil_cloud
        self.behind_cloud = behind_cloud
        self.obstacle_cloud = other_cloud

        if debug:
            print(f"Number of points in remaining cloud: {len(other_points)}")

    def __build_parabola_band_mask(self, H, W, left_right_range, forward_range, left_right_exclusion_range=(-0.2, 0.2)):
        """ 
        Builds a binary mask (H, W) with 1s wherever the cell's physical
        (x, y) coordinate falls between:
            y = 0.4*x^2 + 0     (lower bound)
            y = 0.7*x^2 + 0.5   (upper bound)
        AND the cell's x coordinate is outside exclude_x_range - cells with
        x between exclude_x_range[0] and exclude_x_range[1] are excluded even
        if they'd otherwise fall in the band.
    
        Uses cell-CENTER coordinates (not cell edges), consistent with how
        the occupancy grid itself maps physical space to grid indices.
    
        Assumes grid row = y (forward), grid col = x (left/right), matching
        the rest of the pipeline (x_range=(-2,2), y_range=(0,4)).
        """
        res_x = (left_right_range[1] - left_right_range[0]) / W
        res_y = (forward_range[1] - forward_range[0]) / H
    
        # cell-center physical coordinates for every (row, col)
        row_idx, col_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x = left_right_range[0] + (col_idx + 0.5) * res_x
        y = forward_range[0] + (row_idx + 0.5) * res_y
    
        lower_bound = 0.4 * x**2 + 0
        upper_bound = 0.7 * x**2 + 0.5
    
        mask = (y >= lower_bound) & (y <= upper_bound)
    
        # exclude the central x-slice
        mask &= ~((x >= left_right_exclusion_range[0]) & (x <= left_right_exclusion_range[1]))
    
        return mask

    def __open_occ(self, og, mask, debug, kernel_width=3):
        """
        Cleans a binary top-down occupancy grid, using a precomputed binary
        mask (same shape as og) to target known LiDAR blind-spot regions with
        a vertically-elongated closing operation, without risking any effect
        outside that region.
    
        mask: (H, W) boolean array marking the blind-spot region(s) - e.g.
        built empirically (build_empirical_mask) or from a fitted curve
        (build_parabola_band_mask). Must match og's shape and coordinate
        convention (x_range=(-2,2), y_range=(0,4)).
    
        kernel_width, kernel_height: the special-operation kernel is a tall,
        narrow RECTANGLE (long side along y) rather than an isotropic blob -
        it bridges a pixel at the top edge of the mask to one at the bottom
        edge at the same column, using the mask's own top/bottom boundary as
        the effective curved top/bottom of the operation, without needing to
        reach sideways at all. If kernel_height is left as None, it's set
        automatically from the mask's own maximum vertical span (plus a small
        margin), so it's guaranteed large enough to bridge the mask's tallest
        column regardless of its shape.
    
        Because the result is only ever substituted back in where mask==True,
        increasing kernel_height can never distort anything outside the mask -
        there's no tradeoff to "going too large" the way there was with the
        earlier global/bearing-based approaches.
    
        Pipeline:
        1. General Cartesian pre-closing (MORPH_CLOSE only - safe for thin walls).
        2. Vertical fill, restricted to the mask region only.
        3. Gentle Cartesian smoothing.
        4. Remove small leftover noise blobs (connected components).
        """
        min_blob_size = 6
    
        if og.shape != mask.shape:
            raise ValueError(f"og shape {og.shape} does not match mask shape {mask.shape}")
    
        occupancy_grid = np.copy(og)
        grid_u8 = occupancy_grid.astype(np.uint8) * 255
    
        # --- 1. General pre-closing ---
        pre_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grid_u8 = cv2.morphologyEx(grid_u8, cv2.MORPH_DILATE, pre_kernel)
    
        # --- 2. Vertical fill, restricted to the mask ---
        # NOTE: plain symmetric closing (dilate+erode with the SAME kernel) does
        # NOT reliably bridge gaps between thin (1-2px) features - the erosion
        # step undoes the bridge, since it isn't "thick enough" to survive
        # erosion by a kernel that large. Since the mask guarantees containment
        # regardless of what happens inside it, we dilate aggressively (tall,
        # narrow kernel) and only erode back with a much smaller kernel (just
        # enough to clean up jagged edges), so the bridge actually survives.
        # if kernel_height is None:
        #     col_spans = mask.sum(axis=0)  # vertical extent of the mask per column
        #     kernel_height = int(col_spans.max()) + 4  # small margin
        #     kernel_height += (kernel_height % 2 == 0)  # keep it odd
    
        #     if debug:
        #         print('auto kernel_height:', kernel_height)
        kernel_height = 19
    
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, kernel_height)
        )
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
        vertical_dilated = cv2.dilate(grid_u8, vertical_kernel)
        vertical_result = cv2.erode(vertical_dilated, cleanup_kernel)
    
        masked_result = grid_u8.copy()
        masked_result[mask] = vertical_result[mask]
    
        if debug:
            changed = np.count_nonzero(masked_result[mask] != grid_u8[mask])
            print('masked vertical-fill changed pixels:', changed)
    
        grid_u8 = masked_result
    
        # --- 3. gentle final smoothing (kept as-is) ---
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        final = cv2.morphologyEx(grid_u8, cv2.MORPH_CLOSE, small_kernel)
    
        # --- 4. remove small leftover noise blobs (kept as-is) ---
        binary = (final > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_blob_size:
                cleaned[labels == label_id] = 1
    
        return cleaned

    def __height_occ_grids(self, debug):
        ## EXTRACT POINTS ##
        
        obstacle_points = np.asarray(self.obstacle_cloud.points)

        if debug:
            print(f"Number of Points: {len(self.obstacle_cloud.points)}, Shape: {np.shape(self.obstacle_cloud.points)}\n")

        # EXTRACT COORDINATES
        x = obstacle_points[:,0]
        y = obstacle_points[:,1]

        # Raw LiDAR z values
        z_raw = obstacle_points[:,2]

        # CONVERT TO HEIGHT ABOVE GROUND
        # LiDAR z-axis points DOWN
        # therefore:
        #   larger z = closer to floor
        #   smaller z = taller obstacle

        # Estimate ground level
        ground_z = np.percentile(z_raw, 98)

        # Height above ground
        z = ground_z - z_raw

        ## SETUP GRID ##

        # calculate grid size
        grid_width = int(np.ceil((self.G_X_MAX-self.G_X_MIN) / self.G_RESOLUTION))
        grid_height = int(np.ceil((self.G_Y_MAX - self.G_Y_MIN) / self.G_RESOLUTION))

        if debug:
            print(f"Dimensions of Grid:\nWidth: {grid_width} squares -> {self.G_X_MIN}m to {self.G_X_MAX}m\nHeight: {grid_height} squares -> {self.G_Y_MIN} m to {self.G_Y_MAX} m")

        ## INITIALISE HEIGHT GRID ##

        # fill with NaNs so that any unfilled cell remains a NaN
        height_g = np.full((grid_height, grid_width), np.nan)

        ## CONVERT POINTS TO GRID INDICES ##
        g_x = ((x - self.G_X_MIN) / self.G_RESOLUTION).astype(int)
        g_y = ((y - self.G_Y_MIN) / self.G_RESOLUTION).astype(int)

        ## STORE MAX HEIGHT PER CELL IN GRID ##
        for i in range(len(self.obstacle_cloud.points)):
            xi = g_x[i]
            yi = g_y[i]
            height = z[i]

            # skip invalid indices
            if ((xi < 0) or (xi >= grid_width) or (yi < 0) or (yi >= grid_height)):
                continue
            
            # if the current grid cell is an empty cell, fill it
            if np.isnan(height_g[yi, xi]):
                height_g[yi, xi] = height
            # else replace height only if higher
            else:
                height_g[yi, xi] = max(height_g[yi, xi], height)

        height_g = height_g.T

        valid_mask = (height_g > self.MIN_OBST_HEIGHT) & (height_g < self.MAX_OBST_HEIGHT)
        filtered_heights = np.zeros_like(height_g)   # same 2D shape, filled with 0
        filtered_heights[valid_mask] = height_g[valid_mask]   # only copy over the values that pass

        if debug:
            count_before = np.count_nonzero(~np.isnan(height_g) & (height_g != 0))
            count_after = np.count_nonzero(filtered_heights)
            perc = (count_before-count_after) / count_before * 100
            print(f"\nHeight Points\nBefore Height Filtering: {count_before} points recorded\nAfter Filtering {self.MIN_OBST_HEIGHT}m < H < {self.MAX_OBST_HEIGHT}m: {count_after}\nReduced number of points by {perc:.2f}%")

        ## INITIALISE OCCUPANCY GRID ##
        occupancy_g = np.zeros((grid_height, grid_width), dtype=np.uint32)

        ## FIND CELLS WITH HEIGHTS WRITTEN AND MARK ##
        occupancy_g[valid_mask] = 1

        self.raw_occupancy_grid = occupancy_g

        self.blindspot_m = self.__build_parabola_band_mask(occupancy_g.shape[0], occupancy_g.shape[1], left_right_exclusion_range=(-0.7, 0.7), forward_range=(self.G_X_MIN, self.G_X_MAX), left_right_range=(self.G_Y_MIN, self.G_Y_MAX))
        filtered_occ = self.__open_occ(occupancy_g, self.blindspot_m, debug, kernel_width=5)

        self.height_grid = height_g
        self.occupancy_grid = filtered_occ