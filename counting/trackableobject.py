class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		self.objectID = objectID
		self.centroids = [centroid]

		# used to indicate if the object has
		# already been counted or not
		self.counted = False
