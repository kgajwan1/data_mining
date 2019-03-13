#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <limits>
#include <queue>
#include <utility>
#include <chrono>
#include <iomanip>
#include <thread>
#include <stdlib.h>
#include <mutex>
#include <iterator>
#include <memory>
using namespace std;

// Global Variables
uint64_t NUM_QUERIES = 0;
uint64_t QUERY_DIMS = 0;
uint64_t K = 0;
uint64_t TRAINING_FILE_ID = 0;
uint64_t QUERY_FILE_ID = 0;
uint64_t TRAIN_PTS = 0;
uint64_t TRAIN_DIMS = 0;
unsigned int num_threads = 0;
unsigned int MAX_THREADS = 0;
vector<uint64_t> queries_per_thread;
vector<thread> tree_threads;
mutex mtx;

// STRUCT/CLASS DEFINITIONS
/* --------------------------------------------------------------------- */
struct Point {
		vector<float> coords;
		~Point() {}
};

/* --------------------------------------------------------------------- */
class FixedQueue
{
	private:
		uint64_t fixed_size;
		vector<pair<Point*, float>> neighbors;
	public:
		FixedQueue(uint64_t sz) : fixed_size{sz} {}
		~FixedQueue() {}
		void insert_if_nearer(pair<Point*, float>);
		static bool cmpDist(pair<Point*, float> p1, pair<Point*, float> p2)
		{
			return p1.second < p2.second;
		}
		vector<pair<Point*, float>> get_neighbors() { return neighbors; }
		uint64_t size() { return neighbors.size(); }
		float get_max() { return neighbors[neighbors.size()-1].second; }
		bool not_full() { return neighbors.size() < K; }
};

void FixedQueue::insert_if_nearer(pair<Point*, float> temp)
{
	// If Queue full, check to see if new point is smaller than biggest entry
	if (this->neighbors.size() == this->fixed_size)
	{
		if (temp.second < neighbors[fixed_size-1].second)
		{
			neighbors[fixed_size-1] = temp;
			sort(this->neighbors.begin(), this->neighbors.end(), this->cmpDist);
		}
	}
	// If not full just throw it in there
	else
	{
		this->neighbors.push_back(temp);
		sort(this->neighbors.begin(), this->neighbors.end(), this->cmpDist);
	}
}

/* --------------------------------------------------------------------- */
// Node class for representing the KD-Tree - basically a binary tree
class Node {
    private:
        Point* point;
		Node *left_child;
		Node *right_child;
    public:
		// Constructor
        Node(Point* p) : point{p}, left_child{nullptr}, right_child{nullptr} {}
		~Node() {}
		vector<float> get_coords() { return point->coords; }
		float get_coords(uint64_t i) { return point->coords[i]; }
		Node* get_left() { return left_child; }
		Node* get_right() { return right_child; }
		void insert_left(Node* child) { this->left_child = child; }
		void insert_right(Node* child) { this->right_child = child; }
		void query(Point *q, Node *n, int dim, FixedQueue* cur_neighbors);
};

/* --------------------------------------------------------------------- */
// Implement class to allow passing dimension for comparison in contructor
class cmp 
{
	public:
		cmp(int d) : dim(d) {}
		bool operator()(Point* p1, Point* p2) 
		{
			return p1->coords[dim] < p2->coords[dim];
    	}
	private:
    	int dim;
};

/* --------------------------------------------------------------------- */
/* HELPER FUNCTIONS	*/
void write_output_file(const char* filename, vector<vector<pair<Point*, float>>>* nearest_neighbors_vec)
{
	ofstream outfile (filename, ios::binary); // open in binary mode
	// Invalid file handling
	if(!outfile)
	{
		cout << "No file named " << filename << endl;
		exit(1);
	}

	char filetype[8] = "RESULT";
	ifstream random_file ("/dev/urandom", ios::in|ios::binary); // open in binary mode
	static char buff[8];
	random_file.read(buff, 8);
	uint64_t result_file_ID = *((uint64_t*) buff);

	outfile.write(reinterpret_cast<const char *> (&filetype), sizeof(filetype));
	outfile.write(reinterpret_cast<const char *> (&TRAINING_FILE_ID), sizeof(TRAINING_FILE_ID));
	outfile.write(reinterpret_cast<const char *> (&QUERY_FILE_ID), sizeof(QUERY_FILE_ID));
	outfile.write(reinterpret_cast<const char *> (&result_file_ID), sizeof(result_file_ID));
	outfile.write(reinterpret_cast<const char *> (&NUM_QUERIES), sizeof(NUM_QUERIES));
	outfile.write(reinterpret_cast<const char *> (&TRAIN_DIMS), sizeof(TRAIN_DIMS));
	outfile.write(reinterpret_cast<const char *> (&K), sizeof(K));
	for (uint64_t q = 0; q < NUM_QUERIES; q++)
	{
		for(uint64_t i = 0; i < K; i++)
		{
			for(uint64_t j = 0; j < QUERY_DIMS; j++)
			{
				outfile.write(reinterpret_cast<const char *> (&((*nearest_neighbors_vec)[q][i].first->coords[j])), sizeof(float));
			}
		}
	}
	outfile.close();
}

vector<Point*> read_query_file(char* filename)
{
	static char buff[8] = {}; // Buffer for reading data from file
	ifstream file (filename, ios::in|ios::binary); // open in binary mode

	// Invalid file handling
	if(!file)
	{
		cout << "No file named " << filename << endl;
		exit(1);
	}

	cout << "\n----QUERY_FILE RESULTS----" << endl;
	// Read in header data from file
	file.read(buff, 8);
	string filetype(buff);
	cout << filetype << endl;

	file.read(buff, 8);
	QUERY_FILE_ID = *((uint64_t*) buff);
	cout << "QUERY FILE ID: " << QUERY_FILE_ID << endl;

	file.read(buff, 8);
	NUM_QUERIES = *((uint64_t*) buff);
	cout << "NUM_QUERIES: " << NUM_QUERIES << endl;

	file.read(buff, 8);
	QUERY_DIMS = *((uint64_t*) buff);
	cout << "QUERY_DIMS: " << QUERY_DIMS << endl;
	assert(QUERY_DIMS == TRAIN_DIMS);

	file.read(buff, 8);
	K = *((uint64_t*) buff);
	cout << "K: " << K << endl;
	assert(K > 0);

	// Create vector of size # training points
	vector<Point*> data;

	// For each training point, get each dimension, then add it to vector
	for(uint64_t i = 0; i < NUM_QUERIES; i++)
	{
		Point* temp = new Point();
		for(uint64_t j = 0; j < QUERY_DIMS; j++)
		{
			// Read each 32bit float coordinate dimension
			file.read(buff, 4);
			temp->coords.push_back(*((float*) buff));
		}
		data.push_back(temp);
	}
	file.close();

	cout << "Number of query points in vector: " << data.size() << endl;
	assert(data.size() == NUM_QUERIES);
	return data;
}


shared_ptr<vector<Point*>> read_training_file(char* filename)
{
	static char buff[8] = {}; // Buffer for reading data from file
	ifstream file (filename, ios::in|ios::binary); // open in binary mode

	// Invalid file handling
	if(!file)
	{
		cout << "No file named " << filename << endl;
		exit(1);
	}

	cout << "----TRAINING_FILE RESULTS----" << endl;
	// Read in header data from file
	file.read(buff, 8);
	string filetype(buff);
	cout << filetype << endl;

	file.read(buff, 8);
	string fid(buff);
	TRAINING_FILE_ID = *((uint64_t*) buff);
	cout << "TRAINING_FILE_ID: " << TRAINING_FILE_ID << endl;

	file.read(buff, 8);
	TRAIN_PTS = *((uint64_t*) buff);
	cout << "TRAIN_PTS: " << TRAIN_PTS << endl;

	file.read(buff, 8);
	TRAIN_DIMS = *((uint64_t*) buff);
	cout << "TRAIN_DIMS: " << TRAIN_DIMS << endl;

	// Create vector of size # training points
	shared_ptr<vector<Point*>> data (new vector<Point*>());

	// For each training point, get each dimension, then add it to vector
	for(uint64_t i = 0; i < TRAIN_PTS; i++)
	{
		Point* temp = new Point();
		for(uint64_t j = 0; j < TRAIN_DIMS; j++)
		{
			// Read each 32bit float coordinate dimension
			file.read(buff, 4);
			temp->coords.push_back(*((float*) buff));
		}
		data->push_back(temp);
	}
	file.close();

	cout << "Number of training points in vector: " << data->size() << endl;
	assert(data->size() == TRAIN_PTS);
	return data;
}

// Returns index of median along given dimension
int find_median(shared_ptr<vector<Point*>> subarray, int dimension)
{
	if (subarray->size() > 1)
	{
		// Sort points in array by the values of the current dimension
		sort(subarray->begin(), subarray->end(), cmp(dimension));
	}
	if (subarray->size() % 2 == 0)
	{
		return (subarray->size() / 2) - 1;
	}
	else
	{
		return (subarray->size() - 1) / 2;
	}
}

void build_left_sequential(Node* parent, shared_ptr<vector<Point*>> data, int dim);
void build_right_sequential(Node* parent, shared_ptr<vector<Point*>> data, int dim);

void build_left_sequential(Node* parent, shared_ptr<vector<Point*>> data, int dim)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}

	int median_index = find_median(data, dim);
	assert(median_index >= 0);

	Point* median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as left child to parent
	Node* split = new Node(median_point);
	parent->insert_left(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	build_left_sequential(split, left, (dim+1) % TRAIN_DIMS);
	build_right_sequential(split, right, (dim+1) % TRAIN_DIMS);

}

void build_right_sequential(Node* parent, shared_ptr<vector<Point*>> data, int dim)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}

	int median_index = find_median(data, dim);
	assert(median_index >= 0);
	
	Point* median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as right child to parent
	Node* split = new Node(median_point);
	parent->insert_right(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	build_left_sequential(split, left, (dim+1) % TRAIN_DIMS);
	build_right_sequential(split, right, (dim+1) % TRAIN_DIMS);
}


void build_left(Node* parent, shared_ptr<vector<Point*>> data, int dim);
void build_right(Node* parent, shared_ptr<vector<Point*>> data, int dim);

void build_left(Node* parent, shared_ptr<vector<Point*>> data, int dim)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}

	int median_index = find_median(data, dim);
	assert(median_index >= 0);

	Point* median_point = (*data)[median_index];
	assert(median_point != nullptr);

	// Add median point as left child to parent
	Node* split = new Node(median_point);
	parent->insert_left(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	// Lock mutex so multiple threads don't enter if statement
	unique_lock<mutex> lck(mtx);
	if (num_threads < MAX_THREADS)
	{
		tree_threads.push_back(thread(build_left, split, left, (dim+1) % TRAIN_DIMS));
		num_threads++;
		lck.unlock();
		build_right(split, right, (dim+1) % TRAIN_DIMS);
	}
	else
	{
		lck.unlock();
		build_left_sequential(split, left, (dim+1) % TRAIN_DIMS);
		build_right_sequential(split, right, (dim+1) % TRAIN_DIMS);
	}
}

void build_right(Node* parent, shared_ptr<vector<Point*>> data, int dim)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}

	int median_index = find_median(data, dim);
	assert(median_index >= 0);

	Point* median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as right child to parent
	Node* split = new Node(median_point);
	parent->insert_right(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	// Lock mutex so multiple threads don't enter if statement
	unique_lock<mutex> lck(mtx);
	if (num_threads < MAX_THREADS)
	{
		tree_threads.push_back(thread(build_left, split, left, (dim+1) % TRAIN_DIMS));
		num_threads++;
		lck.unlock();
		build_right(split, right, (dim+1) % TRAIN_DIMS);
	}
	else
	{
		lck.unlock();
		build_left_sequential(split, left, (dim+1) % TRAIN_DIMS);
		build_right_sequential(split, right, (dim+1) % TRAIN_DIMS);
	}
}

// Traverse tree in post-order to delete all nodes
void delete_kd_tree(Node* current)
{
	if (current != nullptr)
	{
		delete_kd_tree(current->get_left());
		delete_kd_tree(current->get_right());
		delete current;
	}
}

// Traverse tree in pre-order to print all nodes
void print_kd_tree(Node* current, string dir, int level)
{
	cout << dir << level << ": ";
	for(uint64_t j = 0; j < TRAIN_DIMS; j++)
	{
		 cout << current->get_coords(j) << "  ";
	}
	cout << endl;

	if (current->get_left() != nullptr)
	{
		print_kd_tree(current->get_left(), "LEFT", level+1);
	}
	if (current->get_right() != nullptr)
	{
		print_kd_tree(current->get_right(), "RIGHT", level+1);
	}
}

float dist(Point* p1, Point* p2)
{
	float total = 0.0;
	float diff = 0.0;
	for (uint64_t i = 0; i < QUERY_DIMS; i++)
	{
		diff = p1->coords[i] - p2->coords[i];
		total += diff*diff;
	}
	return sqrt(total);
}

void Node::query(Point *q, Node *n, int dim, FixedQueue* cur_neighbors) 
{
	if (n == nullptr) 
	{
		return;
	}

	// Index of next dimension to consider (one level down).
	int next_dim = (dim + 1) % TRAIN_DIMS;

	float d = dist(q, n->point);
	cur_neighbors->insert_if_nearer(make_pair(n->point, d));

	if (q->coords[dim] <= n->point->coords[dim]) 
	{
		query(q, n->left_child, next_dim, cur_neighbors);
		// Is the hyperplane closer?
		if (n->point->coords[dim] - q->coords[dim] < cur_neighbors->get_max() ||
			cur_neighbors->not_full()) 
		{
			query(q, n->right_child, next_dim, cur_neighbors);
		}
	} 
	else 
	{
		query(q, n->right_child, next_dim, cur_neighbors);
		// Is the hyperplane closer?
		if (q->coords[dim] - n->point->coords[dim] < cur_neighbors->get_max() ||
			cur_neighbors->not_full()) 
		{
			query(q, n->left_child, next_dim, cur_neighbors);
		}
	}
}

void dispatch_query_threads(int thread_num, vector<Point*>* query_data, 
					Node* root, vector<FixedQueue*>* fq_vec, 
					vector<vector<pair<Point*, float>>>* nearest_neighbors_vec)
{
	uint64_t queries_to_do = queries_per_thread[thread_num];
	uint64_t base = 0;
	for (int i = 0; i < thread_num; i++)
	{
		base += queries_per_thread[i];
	}
	for (uint64_t q = 0; q < queries_to_do; q++)
	{
		uint64_t index = base + q;
		(*fq_vec)[index] = new FixedQueue(K);
		root->query((*query_data)[index], root, 0, (*fq_vec)[index]);
	 	(*nearest_neighbors_vec)[index] = (*fq_vec)[index]->get_neighbors();
	}
}

int main(int argc, char* argv[])
{
	if(argc < 5)
	{
		cout << "Execution format: ./a.out <num_cores> <training_file> <query_file> <results_file>" << endl;
		exit(1);
	}
	// Assign command-line args
	int num_cores = atoi(argv[1]); 
	assert(num_cores > 0);
	MAX_THREADS = num_cores;
	char* training_file = argv[2];
	char* query_file = argv[3];
	char* results_file = argv[4];

	// READ TRAINING FILE
	auto start = chrono::high_resolution_clock::now();

	shared_ptr<vector<Point*>> training_data = read_training_file(training_file);	

	auto stop = chrono::high_resolution_clock::now();
	chrono::duration<double> dt = stop - start;
	cout << "Time for reading training file: " << dt.count() << endl;
	
	// READ QUERY FILE
	start = chrono::high_resolution_clock::now();
	
	vector<Point*> query_data = read_query_file(query_file);
	
	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time for reading query file: " << dt.count() << endl;
	
	// CHECKING TRAINING FILE RESULTS
	assert(training_data->size() == TRAIN_PTS);

	// Build K-D Tree
	start = chrono::high_resolution_clock::now();
	
	int median_index = find_median(training_data, 0);
	Point* median_point = (*training_data)[median_index];
	Node* root = new Node(median_point);
	
	shared_ptr<vector<Point*>> left (new vector<Point*>(training_data->begin(), training_data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(training_data->begin()+median_index+1, training_data->end()));

	// Call subfunctions to recursively split each half of the training_data
	num_threads = 1;
	tree_threads.push_back(thread(build_left, root, left, 1 % TRAIN_DIMS));
	build_right(root, right, 1 % TRAIN_DIMS);
	
	// Handle cleaning up the threads
	cout << "\n----BUILDING K-D TREE----" << endl;

	while (num_threads != MAX_THREADS)
	{
		continue;
	}
	while (num_threads != tree_threads.size())
	{
		continue;
	}

	cout << "All threads created!" << endl;
	cout << "Joining Tree Threads now..." << endl;

	for (unsigned int i = 0; i < MAX_THREADS; i++)
	{
		//cout << "Joining thread " << i+1 << "/" << MAX_THREADS << endl;
		if (tree_threads[i].joinable())
		{
			tree_threads[i].join();
		}
	}
	cout << "All threads joined!" << endl;

	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time to build KD-Tree: " << dt.count() << endl;

	cout << "\n----QUERYING TREE----" << endl;
	// CHECKING FILE DATA
	cout << "Number of query points in vector: " << query_data.size() << endl;
	assert(query_data.size() == NUM_QUERIES);
	
	// Priority Queue of pairs based on comparator for distances in pair
	vector<FixedQueue*> fq_vec(NUM_QUERIES);
	// vec of vectors of nearest neighbors
	vector<vector<pair<Point*, float>>> nearest_neighbors_vec(NUM_QUERIES);


	// Get start time for all queries
	start = chrono::high_resolution_clock::now();

	// IF MORE THREADS THAN QUERIES, ONLY MAKE 1 THREAD FOR EACH
	if (MAX_THREADS > NUM_QUERIES)
	{
		MAX_THREADS = NUM_QUERIES;
	}

	// Initialize to most even distribution of queries
	queries_per_thread = vector<uint64_t>(MAX_THREADS, (int) NUM_QUERIES/MAX_THREADS);
	// Get remainder of the division
	uint64_t temp = NUM_QUERIES % MAX_THREADS;
	unsigned int i = 0;
	// Add 1 to each thread's workload until remainder is gone
	while (temp != 0)
	{
		queries_per_thread[i]++;
		temp--;
		i++;
	}

	start = chrono::high_resolution_clock::now();
 
	vector<thread> query_threads;
	for (i = 0; i < MAX_THREADS; i++)
	{
		query_threads.push_back(thread(dispatch_query_threads, i, &query_data, root, &fq_vec, &nearest_neighbors_vec));
		this_thread::sleep_for(chrono::milliseconds(1));
	}

	for (i = 0; i < MAX_THREADS; i++)
	{
		query_threads[i].join(); // Wait until thread is done
	}
	
	stop = chrono::high_resolution_clock::now();
	dt = stop - start;

// PRINT OUT ALL NEAREST NEIGHBORS FOR EACH QUERY
/*	for (uint64_t q = 0; q < NUM_QUERIES; q++)
	{
		cout << "\n----NEAREST NEIGHBORS for query" << q+1 << "----\n" << endl;
		cout << "Number of nearest neighbors actually found: " << nearest_neighbors_vec[q].size() << endl;
		cout << "Number of nearest neighbors supposed to be found: " << K << endl;
		cout.precision(3);
		cout << fixed;
		//for(uint64_t i = 0; i < K; i++)
		for(uint64_t i = 0; i < K; i++)
		{
			for(uint64_t j = 0; j < QUERY_DIMS; j++)
			{
				cout << nearest_neighbors_vec[q][i].first->coords[j] << "   ";
			}
			cout << "Distance: " << nearest_neighbors_vec[q][i].second << endl;
		}
	}
*/	

	cout << "Time to query KD-Tree for all queries: " << dt.count() << endl;

	cout << "\n----RESULTS FILE----" << endl;
	cout << "Writing results file..." << endl;

	start = chrono::high_resolution_clock::now();

	write_output_file(results_file, &nearest_neighbors_vec);

	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time to write results file: " << dt.count() << endl;


	// CLEAN UP REMAINING POINTERS
	cout << "\n----CLEANING UP----" << endl;
	cout << "Deleting training data..." << endl;
	for (uint64_t i = 0; i < TRAIN_PTS; i++)
	{
		delete (*training_data)[i];
	}

	cout << "Deleting query and nearest neighbor data..." << endl;
	for (uint64_t i = 0; i < NUM_QUERIES; i++)
	{
		delete fq_vec[i];
delete
 query_data[i];
	}

	cout << "Deleting tree data...\n" << endl;
	delete_kd_tree(root);

	return 0;
}
