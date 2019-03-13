#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <limits>
#include <queue>

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <vector>

#include <thread>
#include <stdlib.h>
#include <mutex>
#include <iterator>
#include <memory>

#include <utility>
#include <iomanip>

using namespace std;
using namespace std :: chrono;

// uint64_t  credited to sir on his mail
// Global Variables sue me

uint64_t NUM_QUERIES = 0;
uint64_t QUERY_DIMS = 0;
uint64_t K = 0;
unsigned int num_threads = 0;
unsigned int MAX_THREADS = 0;
vector<uint64_t> num_queries_per_thread;
vector<thread> threads_tree;
uint64_t TF_ID= 0;
uint64_t QF_ID= 0;
uint64_t TRAIN_PTS = 0;
uint64_t TRAIN_DIMS = 0;
mutex mut_x;

/*
wish to integrate this code in read file...
//os code for creating and parsing data through dev/numpipe

Course: Operating Systems
Assignment 3: Character Device

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <asm/uaccess.h>
#include <linux/slab.h>
#include <linux/semaphore.h>
#include <linux/init.h>
#include <linux/miscdevice.h>
#include <linux/platform_device.h>
#include <linux/rtc.h>
#include <linux/sched.h>

/*defining author, license and description of this module*/
MODULE_LICENSE("GPL");
MODULE_AUTHOR("GAJJU");
MODULE_DESCRIPTION("NUMPIPE");

/*defining sempaphore for buffer*/
static struct semaphore full;
static struct semaphore empty;
static struct semaphore read_op_mutex;
static struct semaphore write_op_mutex;
/*misc device struct*/
static struct miscdevice my_device;
/*number of open devices*/
static int open_count;
/*buffer size to store the characters*/
static int buffer_size;

static int string_char_count = 10000;//string size too
static int read_index = 0, write_index = 0;
/*need at least 1 empty slot to READ/write*/
static int buffer_empty_slots;

/*getting buffer size from command line*/
module_param(buffer_size, int, 0000);

/*buffer to store strings*/
char** buffer;

/*declaring fops functions*/
static ssize_t my_write(struct file*, const char*, size_t, loff_t*);
static ssize_t my_read(struct file*, char*, size_t, loff_t*);
static int my_release(struct inode*, struct file*);
static int my_open(struct inode*, struct file*);

static struct file_operations my_device_fops = {
	.open = &my_open,
	.read = &my_read,
	.write = &my_write,
	.release = &my_release
};

int init_module(){
	/*initializing parameters*/
	
	my_device.minor = MISC_DYNAMIC_MINOR;
	my_device.fops = &my_device_fops;
	
	/*registering the device*/
	int register_return_value;
	if((register_return_value = misc_register(&my_device))){
		printk(KERN_ERR "Unable to register the device\n");
		return register_return_value;
	}
        else{
	printk(KERN_INFO "Device Registered!\n");
	printk(KERN_INFO "Details\n--------------\n");
	printk(KERN_INFO "Name: numpipe\n");
	printk(KERN_INFO "Queue size: %d\n", buffer_size);
	}
	/*allocating memory for the buffer. 2d array*/
	int _allocated = 0;
	buffer = (char**)kmalloc(buffer_size*sizeof(char*), GFP_KERNEL);
	while(_allocated < buffer_size){
		buffer[_allocated] = (char*)kmalloc((string_char_count+1)*sizeof(char), GFP_KERNEL);
		buffer[string_char_count] = '\0';
		++_allocated;
	}
	sema_init(&full, 0);
	sema_init(&empty, buffer_size);
	sema_init(&read_op_mutex, 1);
	sema_init(&write_op_mutex, 1);
	buffer_empty_slots = buffer_size;
	open_count = 0;
	return 0;
}

/*cleanup the module*/
void cleanup_module(){
	/*freeing memory*/
	int _iter;
	for(_iter = 0; _iter < buffer_size; _iter++){
		kfree(buffer[_iter]);
	}
	misc_deregister(&my_device);
	printk(KERN_INFO "Numpipe Unregistered!\n");
}

static int my_open(struct inode* _inode, struct file* _file){
	
	++open_count;
	return 0;
}

static int my_release(struct inode* _inode, struct file* _file){
	--open_count;
	return 0;
}

static ssize_t my_read(struct file* _file, char* user_buffer, size_t chars_to_be_read, loff_t* offset){
	int user_queue_index = 0;
	down_interruptible(&read_op_mutex);
	down_interruptible(&full);
	read_index %= buffer_size;
	for(user_queue_index = 0; user_queue_index < chars_to_be_read; user_queue_index++){
		if(buffer_empty_slots >= buffer_size){
			break;
		}
		copy_to_user(&user_buffer[user_queue_index], &buffer[read_index][user_queue_index], 1);
	}
	++read_index;
	++buffer_empty_slots;
	up(&empty);
	up(&read_op_mutex);
	return user_queue_index;
}
static ssize_t my_write(struct file* _file, const char* user_buffer, size_t number_of_chars_to_write, loff_t* offset){
	int user_queue_index = 0;
	int i = 0;
	
	down_interruptible(&write_op_mutex);
	down_interruptible(&empty);
	write_index %= buffer_size;
	for(user_queue_index = 0; user_queue_index < number_of_chars_to_write; user_queue_index++){
		if(buffer_empty_slots <= 0){
			break;
		}
		copy_from_user(&buffer[write_index][user_queue_index], &user_buffer[user_queue_index], 1);
	}
	++write_index;
	--buffer_empty_slots;
	up(&full);
	up(&write_op_mutex);
	return user_queue_index;
}

*/

// NO VECTOR OF VECTORS, I SUCK AT THAT, MAKE POINTS NOT VECTORS  //SLOGAN 101
struct Point 
{
		vector<float> coordinates;
		~Point() {}
};

//QUEUE TO HELP WITH KNN
class Queue
{
	public:
		Queue(uint64_t sz) : fixed_size{sz} {}
		~Queue() {}
		
		//stanford notes referred for help in priority queue
		void insert_if_nearer(pair<Point*, float>);
		
		static bool compareDistance(pair<Point*, float> p1, pair<Point*, float> p2)
		{
			//figured out this is better than if else
			return p1.second < p2.second;
		}

		bool full_checker() 
		{ 
			//stop when you have found enough neighbors
			return neighbours.size() < K; 
		}

		vector<pair<Point*, float>> get_neighbours() 
		{ 
			return neighbours; 
		}

		uint64_t size() 
		{ 
			return neighbours.size(); 
		}

		float get_max() 
		{
			return neighbours[neighbours.size()-1].second; 
		}

		//no modi-fying these values
	private:
		uint64_t fixed_size;
		vector<pair<Point*, float>> neighbours;
		
};

void Queue::insert_if_nearer(pair<Point*, float> temp)
{
	// If Queue full, check to see if new point is smaller than biggest entry
	if (this->neighbours.size() == this->fixed_size  &&  temp.second < neighbours[fixed_size-1].second)
	{
		neighbours[fixed_size-1] = temp;
		sort(this->neighbours.begin(), this->neighbours.end(), this->compareDistance);
		
	}
	// If not full just throw it in there
	else
	{
		this->neighbours.push_back(temp);
		sort(this->neighbours.begin(), this->neighbours.end(), this->compareDistance);
	}
}

// k-dimension tree, cmu notes

class Node {
    //make it accessible
    public:
		
        Node(Point* p) : point{p}, left_child{nullptr}, right_child{nullptr} {}
		~Node() {}
		vector<float> get_coordinates() 
		{ 	
			return point->coordinates; 
		}
		float get_coordinates(uint64_t i) 
		{
			return point->coordinates[i]; 
		}
	
		Node* get_right() 
		{ 
			return right_child; 
		}
		
		Node* get_left() 
		{ 
			return left_child; 
		}
	
		void insert_right(Node* child) 
		{ 
			this->right_child = child;
		}
		void insert_left(Node* child) 
		{ 
			this->left_child = child; 
		}
		
		void query(Point *q, Node *n, int dimension, Queue* cur_neighbours);

	
	//aint nobody gonna access that 
	private:
        	Point* point;
			Node *left_child;
			Node *right_child;
};

// Implement class to allow passing dimension to compare

class compare 
{
	public:
		compare(int d) : dimension(d) {}
		bool operator()(Point* p1, Point* p2) 
		{
			return p1->coordinates[dimension] < p2->coordinates[dimension];
    	}
	private:
    	int dimension;
};

// Some useful tools

//start with reading training file
shared_ptr<vector<Point*>> read_training_file(char* filename)
{
	static char buff[8] = {}; // Buffer for reading data from file
	ifstream file (filename, ios::in|ios::binary); // open in binary mode

	// Invalid file handling
	if(!file)
	{
		cout << "No file named " << filename << "exists"<< endl;
		exit(1);
	}

	cout << "_______TRAINING_FILE RESULTS______" << endl;
	// Read in header data from file, 8*8 == 64
	file.read(buff, 8);
	string filetype(buff);
	//filetype is written up top, no need to do it twice

	file.read(buff, 8);
	string fid(buff);
	TF_ID= *((uint64_t*) buff);
	cout << "TRAINING_FILE_ID: " << TF_ID<< endl;

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
			// 8 * 4 == 32
			file.read(buff, 4);
			temp->coordinates.push_back(*((float*) buff));
		}
		data->push_back(temp);
	}
	file.close();

	cout << "Number of training points in vector: " << data->size() << endl;
	assert(data->size() == TRAIN_PTS);
	return data;
}


//then you read query
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

	cout << "\n_____QUERY_FILE RESULTS____" << endl;
	// Read in header data from file
	file.read(buff, 8);
	string filetype(buff);
	cout << filetype << endl;

	file.read(buff, 8);
	QF_ID= *((uint64_t*) buff);
	cout << "QUERY FILE ID: " << QF_ID<< endl;

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

	// For each training point, get dimension, then add it to vector
	for(uint64_t i = 0; i < NUM_QUERIES; i++)
	{
		Point* temp = new Point();
		for(uint64_t j = 0; j < QUERY_DIMS; j++)
		{
			// Read each 32bit float coordinate dimension
			file.read(buff, 4);
			temp->coordinates.push_back(*((float*) buff));
		}
		data.push_back(temp);
	}
	file.close();

	cout << "Number of query points in vector: " << data.size() << endl;
	assert(data.size() == NUM_QUERIES);
	return data;
}

//write output
void write_output_file(const char* filename, vector<vector<pair<Point*, float>>>* nearest_neighbours_vec)
{
	ofstream outfile (filename, ios::binary); // open in binary mode
	// file handler if file is invalid
	if(!outfile)
	{
		cout << "Not gonna write there... " << filename << endl;
		exit(1);
	}

	char filetype[8] = "RESULT";
	//no retyping at cout
	ifstream random_file ("/dev/urandom", ios::in|ios::binary); // open in binary mode
	static char buff[8];
	random_file.read(buff, 8);
	uint64_t result_file_ID = *((uint64_t*) buff);

	outfile.write(reinterpret_cast<const char *> (&TF_ID), sizeof(TF_ID));
	outfile.write(reinterpret_cast<const char *> (&QF_ID), sizeof(QF_ID));

	outfile.write(reinterpret_cast<const char *> (&filetype), sizeof(filetype));
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
				outfile.write(reinterpret_cast<const char *> (&((*nearest_neighbours_vec)[q][i].first->coordinates[j])), sizeof(float));
			}
		}
	}
	outfile.close();
}

// Returns index of median along given dimension
int find_median(shared_ptr<vector<Point*>> subarray, int dimension_median)
{
	if (subarray->size() > 1)
	{
		// Sort points in array by the values of the current dimension
		sort(subarray->begin(), subarray->end(), compare(dimension_median));
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

void build_left_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension);
void build_right_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension);
void build_left(Node* parent, shared_ptr<vector<Point*>> data, int dimension);
void build_right(Node* parent, shared_ptr<vector<Point*>> data, int dimension);


void build_left_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}
	
	int median_index;
	Point* median_point;
	Node* split;

	median_index = find_median(data, dimension);
	assert(median_index >= 0);

	median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as left child
	split = new Node(median_point);
	parent->insert_left(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMS);
	build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMS);

}

void build_right_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}
	
	int median_index;
	Point* median_point;
	Node* split;

	median_index = find_median(data, dimension);
	assert(median_index >= 0);
	
	median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as right child
	split = new Node(median_point);
	parent->insert_right(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMS);
	build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMS);
}


void build_left(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}
	
	int median_index;
	Point* median_point;
	Node* split;

	median_index = find_median(data, dimension);
	assert(median_index >= 0);

	median_point = (*data)[median_index];
	assert(median_point != nullptr);

	// Add median point as left child to parent
	split = new Node(median_point);
	parent->insert_left(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	// Lock mutex so multiple threads don't enter if statement
	unique_lock<mutex> lck(mut_x);
	if (num_threads < MAX_THREADS)
	{
		threads_tree.push_back(thread(build_left, split, left, (dimension+1) % TRAIN_DIMS));
		num_threads++;
		lck.unlock();
		build_right(split, right, (dimension+1) % TRAIN_DIMS);
	}
	else
	{
		lck.unlock();
		build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMS);
		build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMS);
	}
}

void build_right(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}
	
	int median_index;
	Point* median_point;
	Node* split;

	median_index = find_median(data, dimension);
	assert(median_index >= 0);

	median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as right child
	split = new Node(median_point);
	parent->insert_right(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin() + median_index+1, data->end()));

	// Lock mutex to prevent access to critical region 
	unique_lock<mutex> lck(mut_x);
	if (num_threads < MAX_THREADS)
	{
		threads_tree.push_back(thread(build_left, split, left, (dimension+1) % TRAIN_DIMS));
		num_threads++;
		lck.unlock();
		build_right(split, right, (dimension+1) % TRAIN_DIMS);
	}
	
	else
	{
		lck.unlock();
		build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMS);
		build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMS);
	}
}

// Traverse tree to delete all nodes
void delete_kd_tree(Node* current)
{
	if (current != nullptr)
	{
		delete_kd_tree(current->get_left());
		delete_kd_tree(current->get_right());
		delete current;
	}
}

// Traverse tree to print nodes
void print_kd_tree(Node* current, string dir, int level)
{
	cout << dir << level << ": ";
	for(uint64_t j = 0; j < TRAIN_DIMS; j++)
	{
		 cout << current->get_coordinates(j) << "  ";
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

float distance(Point* p1, Point* p2)
{
	float sum = 0.0;
	float difference = 0.0;
	for (uint64_t i = 0; i < QUERY_DIMS; i++)
	{
		difference = p1->coordinates[i] - p2->coordinates[i];
		sum += difference*difference;
	}
	return sqrt(sum);
}

void Node::query(Point *q, Node *n, int dimension, Queue* cur_neighbours) 
{
	if (n == nullptr) 
	{
		return;
	}

	// Index of next dimension to consider (one level down).
	int next_dim = (dimension + 1) % TRAIN_DIMS;

	float d = distance(q, n->point);
	cur_neighbours->insert_if_nearer(make_pair(n->point, d));

	if (q->coordinates[dimension] <= n->point->coordinates[dimension]) 
	{
		query(q, n->left_child, next_dim, cur_neighbours);
		
		if (n->point->coordinates[dimension] - q->coordinates[dimension] < cur_neighbours->get_max() ||
			cur_neighbours->full_checker()) 
		{
			query(q, n->right_child, next_dim, cur_neighbours);
		}
	} 

	else 

	{
		query(q, n->right_child, next_dim, cur_neighbours);
		// Is the hyperplane closer?
		if (q->coordinates[dimension] - n->point->coordinates[dimension] < cur_neighbours->get_max() ||
			cur_neighbours->full_checker()) 
		{
			query(q, n->left_child, next_dim, cur_neighbours);
		}
	}
}

void dispatch_query_threads(int thread_num, vector<Point*>* query_data,	Node* root, vector<Queue*>* fq_vec, 
						vector<vector<pair<Point*, float>>>* nearest_neighbours_vec)
{
	uint64_t queries_to_do = num_queries_per_thread[thread_num];
	uint64_t base = 0;
	for (int i = 0; i < thread_num; i++)
	{
		base += num_queries_per_thread[i];
	}
	for (uint64_t q = 0; q < queries_to_do; q++)
	{
		uint64_t index = base + q;
		(*fq_vec)[index] = new Queue(K);
		root->query((*query_data)[index], root, 0, (*fq_vec)[index]);
	 	(*nearest_neighbours_vec)[index] = (*fq_vec)[index]->get_neighbours();
	}
}

int main(int argc, char* argv[])
{
	if(argc != 5)
	{
		cout << "Execution format: ./a.out <num_cores> <training_file> <query_file> <results_file>" << endl;
		cout << "Error"<< endl; 
		exit(1);
	}
	// Assign command-line args
	int num_cores = atoi(argv[1]); 
	assert(num_cores > 0);
	//I had done 2* earlier
	//thanks dushyant
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
	threads_tree.push_back(thread(build_left, root, left, 1 % TRAIN_DIMS));
	build_right(root, right, 1 % TRAIN_DIMS);
	
	// Handle cleaning up the threads
	cout << "\n____BUILDING K-DIMENSION TREE____" << endl;

	while (num_threads != MAX_THREADS && num_threads != threads_tree.size())
	{
		continue;
	}
	
	cout << "All threads created!" << endl;
	cout << "Joining Threads now..." << endl;

	for (unsigned int i = 0; i < MAX_THREADS; i++)
	{
		//cout << "Joining thread " << i+1 << "/" << MAX_THREADS << endl;
		if (threads_tree[i].joinable())
		{
			threads_tree[i].join();
		}
	}

	//jedi, I am
	cout << "All joined, threads are" << endl;

	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time to build Tree: " << dt.count() << endl;

	cout << "\n____QUERYING TREE FOR K NEAREST NEIGHBOURS____" << endl;
	// CHECKING FILE DATA
	cout << "Number of query points in vector: " << query_data.size() << endl;
	assert(query_data.size() == NUM_QUERIES);

	// vec of vectors. Just hate doing this, but No dynamic arrays now
	vector<vector<pair<Point*, float>>> nearest_neighbours_vec(NUM_QUERIES);	

	// Priority Queue of pairs 
	vector<Queue*> fq_vec(NUM_QUERIES);
	
	// Get start time for all queries
	start = chrono::high_resolution_clock::now();

	// IF MORE THREADS THAN QUERIES, ONLY MAKE 1 THREAD FOR EACH
	if (MAX_THREADS > NUM_QUERIES)
	{
		MAX_THREADS = NUM_QUERIES;
	}

	// Initialize to most even distribution of queries
	num_queries_per_thread = vector<uint64_t>(MAX_THREADS, (int) NUM_QUERIES/MAX_THREADS);
	// Get remainder of the division
	uint64_t temp = NUM_QUERIES % MAX_THREADS;
	unsigned int i = 0;
	// Add 1 to each thread's workload until remainder is gone
	while (temp != 0)
	{
		num_queries_per_thread[i]++;
		temp--;
		i++;
	}

	start = chrono::high_resolution_clock::now();
 
	vector<thread> query_threads;
	for (i = 0; i < MAX_THREADS; i++)
	{
		query_threads.push_back(thread(dispatch_query_threads, i, &query_data, root, &fq_vec, &nearest_neighbours_vec));
		this_thread::sleep_for(chrono::milliseconds(1));
		//this_thread::sleep_for(chrono::milliseconds(2));
		//this_thread::sleep_for(chrono::milliseconds(3));
	}

	for (i = 0; i < MAX_THREADS; i++)
	{
		query_threads[i].join(); // Wait until thread is done
	}
	
	stop = chrono::high_resolution_clock::now();
	dt = stop - start;

	cout << "Time to query KD-Tree for all queries: " << dt.count() << endl;

	cout << "\n____RESULTS FILE____" << endl;
	cout << "Writing results file..." << endl;

	start = chrono::high_resolution_clock::now();

	write_output_file(results_file, &nearest_neighbours_vec);

	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time to write results file: " << dt.count() << endl;

	// CLEAN UP REMAINING POINTERS
	cout << "\n____CLEANING UP THE BUILT STUFF____" << endl;
	cout << "Deleting training data..." << endl;
	for (uint64_t i = 0; i < TRAIN_PTS; i++)
	{
		delete (*training_data)[i];
	}

	cout << "DELETING QUERY AND KNN DATA..." << endl;
	for (uint64_t i = 0; i < NUM_QUERIES; i++)
	{
		delete fq_vec[i];
		delete query_data[i];
	}

	cout << "DELETING ENTIRE TREE DATA ...\n" << endl;
	delete_kd_tree(root);
	return 0;

}
