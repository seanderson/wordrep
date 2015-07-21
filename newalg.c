#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000 // MAX = sentence-chars + newline + 0-byte

#define MAX_CODE_LENGTH 40
#define MAX_INDEX_BUFF_SIZE 10 // SEA

const int vocab_hash_size = 30000000;
long long vocab_max_size = 2000000; //this variable must be greater than the size of the vocabulary that is being read
long long layer1_size = 0;//activation node layer size
long long vocab_size = 0;
int target_index = 5; //target word in sentence that will be propagated, default is 5
typedef float real;

struct vocab_word { 
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct vocab_word *vocab;
int *vocab_hash;
char *index_buff;
//network elements imported from file
real *syn1;
real *syn0;
real *neu1;
real *expTable;

//static buffers for building the vocab_word structure array "*vocab"
long long cn_temp = 0;
char word_temp[MAX_STRING];
char code_temp[MAX_CODE_LENGTH];
int point_temp[MAX_CODE_LENGTH], window = 1; 
char codelen_temp = '\000';

//read the syn0 array
void read_syn0(){
  FILE *file_input;
  int i, ctr = 0;
  file_input = fopen("syn0", "r");

  if(NULL == file_input){
    printf("Unable to open the syn0 file\n");
    exit(-1);
  }
  printf("Reading syn0\n");  
  fread(syn0,sizeof(real),vocab_size*layer1_size,file_input);
  fclose(file_input);
  return;
}

//read the syn1 array
void read_syn1(){
  FILE *file_input;
  int i, ctr = 0;
  file_input = fopen("syn1", "r");
  printf("Reading syn1\n");
  if(NULL == file_input){
    printf("Unable to open the syn1 file\n");
    exit(-1);
  }
  fread(syn1,sizeof(real),vocab_size*layer1_size,file_input);

  printf("\n");
  fclose(file_input);
  return;
}

void ReadWordFromFile(FILE * fp){  // read a word structure from the binary 
                                   // vocabulary file, used in BuildVocabFromFile, uncomment lines to check correct import of words
  int i = 0;
  fread(word_temp,sizeof(char)*MAX_STRING,1,fp);
  long offset = ftell(fp);
//  printf("%s","Read string stream ");
//  printf("%ld\n",offset);
  for(i=0; i<MAX_CODE_LENGTH; i++){
    fread(&code_temp[i],sizeof(char),1,fp);  
  }
  offset = ftell(fp);
//  printf("%s","Read code stream ");
//  printf("%ld\n",offset);
  for(i=0; i<MAX_CODE_LENGTH; i++){
    fread(&point_temp[i],sizeof(int),1,fp);
  }
  offset = ftell(fp);
//  printf("%s","Read point stream ");
//  printf("%ld\n",offset);
  fread(&cn_temp,sizeof(long long),1,fp);
  offset = ftell(fp);
//  printf("%s","Read cn stream ");
//  printf("%ld\n",offset);
  fread(&codelen_temp,sizeof(char),1,fp);
  offset = ftell(fp);
//  printf("%s","Read codelen stream ");
//  printf("%ld\n",offset);
}
//constructs the vocab_word structure array "*vocab" from the binary file
void BuildVocabFromFile(){
  int i,j,k = 0; 
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  for (i = 0; i < vocab_size; i++) {
    vocab[i].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[i].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
  FILE *file = fopen("vocab.bin","rb");
  if(file == NULL) printf("Unable to open the vocab.bin file\n");
  for(i=0; i<vocab_size; i++){
      ReadWordFromFile(file);
      unsigned int length = strlen(word_temp)+1;
      vocab[i].word=(char *)calloc(length, sizeof(char));
      strcpy(vocab[i].word,word_temp);
      for(j=0; j<MAX_CODE_LENGTH;j++) vocab[i].code[j] = code_temp[j];
      for(j=0; j<MAX_CODE_LENGTH;j++) vocab[i].point[j] = point_temp[j];
      vocab[i].cn = cn_temp;
      vocab[i].codelen = codelen_temp;
     /* printf("\n");
      k = ftell(file);
      printf("%d\n",k);
      printf("\n");*///uncomment to keep track of vocab import
    }
  fclose(file);
}
//constructs the vocabulary hash from the file it was stored in
void BuildVocabHashFromFile(){
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  FILE* hash_file = fopen("vocab_hash","rb");
  fread(vocab_hash,sizeof(int)*vocab_hash_size,1,hash_file);
  fclose(hash_file);
}
//returns hash value of a word, imported from word2vec
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}
//returns word index, called by ReadWordIndex
SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

/*
  Read next word from file.
 */
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

/*
  Returns the word index by calling SearchVocab.
*/
int ReadWordIndex(FILE *fin) { //imported from word2vec
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

/*
  return wordcount of current line for file stream fi.
  Assumes words are space-delimited and each line is newline delimited.
*/
int LineWordCount(char *sent) {
  int i=0;
  int nwords = 0; //length of current line
  char c;
  int nchar = strlen(sent); // "word1 word2 ... \n\0"
  int inword = 0; // true if cursor is in a word.

  while (i < nchar) {
    c = sent[i];
    if ( c == '\n') break;
    if (!inword && !isspace(c)) { // change of state from space to a word IS a new word
      nwords++;
      inword = 1;
      //printf("word onset %c\n",c);
    } 
    else if (inword && isspace(c)) { // simply the end of current word
      inword = 0;
    }
    i++;
  }
  return nwords;
}

/*
  Converts a line in a file from text to an array of vocabulary
  indices, the input form for forward propagation.
*/
long long * FileToSen(int length, FILE* fi){ 

  long long *sen = (long long *) calloc(MAX_SENTENCE_LENGTH,sizeof(long long));
  long long word = 0,sentence_length = 0;
  if (sentence_length == 0) {
    while (1) {
      word = ReadWordIndex(fi);
      if (feof(fi)) break;
      if (word == -1) continue;
      if (word == 0) break;
      
      sen[sentence_length] = word;
      sentence_length++;
      if (sentence_length >= length) break;
    }
   return sen;
  }
}
//performs forward propagation and returns the product of probabilites given the target index, window and indexed word sentence pointer
long double ForwardPropagate(int length,long long* sen){
  int i,j = 0; //iterator
  //allocating memory for the temporary network that will be used in this function
  real *neu1_temp = (real*)calloc(layer1_size,sizeof(real));
  real *syn0_temp = (real*)calloc(layer1_size*vocab_size,sizeof(real));
  real *syn1_temp = (real*)calloc(layer1_size*vocab_size,sizeof(real));
  //copying the network imported to file into the temporary network that will be used for forward propagation
  for(i=0;i<layer1_size;i++) neu1_temp[i]=0;
  for(i=0;i<layer1_size*vocab_size;i++) syn0_temp[i]=syn0[i];
  for(i=0;i<layer1_size*vocab_size;i++) syn1_temp[i]=syn1[i];
  //initializing variables used for forward propagation
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  sentence_position = target_index;
  sentence_length = length;
  long long l1, l2, c;
  char * blabla;
  long double f; 
  long double fbuffer[MAX_CODE_LENGTH]; //storage for all the f's that will be multiplied
  long double result = 1; //result that will be returned

  unsigned long long next_random = 1; //this is assigned the thread ID number in word2vec(during training), we're using a single thread thus this is 1

  word = sen[sentence_position];
  next_random = next_random * (unsigned long long)25214903917 + 11;
  b = next_random % window;
  // in -> hidden
  cw = 0;
  for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
    c = sentence_position - window + a;
    if (c < 0) continue;
    if (c >= sentence_length) continue;
    last_word = sen[c];
    if (last_word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1_temp[c] += syn0_temp[c + last_word * layer1_size];
    cw++;
  }
  if (cw) {
    for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
    for (d = 0; d < vocab[word].codelen; d++) {
      f = 0;
      l2 = vocab[word].point[d] * layer1_size;
      // Propagate hidden -> output
      for (c = 0; c < layer1_size; c++) f += neu1_temp[c] * syn1_temp[c + l2];
      printf("%s","1: ");
      printf("%Le\n",f);
      if (f <= -MAX_EXP){
       fbuffer[d]=1;
        continue;
      }
      else if (f >= MAX_EXP){
        fbuffer[d]=1;
        continue;
      }
      else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
      printf("%s","2: ");
      printf("%Le\n",f);
      fbuffer[d]=f;
    } printf("\n");
  }
  for(j = 0; j < vocab[word].codelen; j++) result = result * fbuffer[j];

  free(neu1_temp);
  free(syn0_temp);
  free(syn1_temp);

   return result;
}
//imported from word2vec and used for reading command line arguments
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

void ReadIndexFromFile(FILE *indices){
  char ch;
  int i = 0;
  while (i < MAX_INDEX_BUFF_SIZE && !feof(indices) ){ // SEA added feof test
    ch = fgetc(indices);
    if(ch == '\n') break;
    else {
      index_buff[i]=ch;
      i++;
    }
  }
}
int GetIndex(){
  int rtn_item = 0;
  if(index_buff[0]=='0'){
    rtn_item = index_buff[1];
    return rtn_item - '0';
  } 
  else{
    rtn_item = (index_buff[0]-'0')*10 + (index_buff[1]-'0');
    return rtn_item;
  }
}

int Lines(FILE* fp){
  int lines = 0;
  char ch;
  while(!feof(fp))
  {
    ch = fgetc(fp);
    if(ch == '\n')
    {
      lines++;
    }
  }
  return lines;
}

/*
  Read exactly one line into line.
  Return 1 if successful and -1 if EOF or error.
  line is empty if no chars read.
*/
int readLine(FILE *fp, char *line) {

    // Read one line and store for repeated use.  line will be "sentence chars\n\0"
    line[0] = '\0';
    if (fgets(line,MAX_SENTENCE_LENGTH,fp) == NULL || strlen(line) >= MAX_SENTENCE_LENGTH - 1) {
      if (strlen(line) == 0) return(1); // normal exit on EOF
      else {
  	fprintf(stderr,"ERROR: Line len %d is too long: %s",(int) strlen(line),line);
  	return (-1);
      }
    }
    return (1);
}



int main(int argc, char **argv) {
  int i,j,k = 0;//counters
  if(argc == 1){  //printing instructions
    printf("\n");
    printf("Forward propagation of sentences in a file delimited by \\n\n\n");
    printf("Parameters:\n");
    printf("\tValue for the vocabulary size that resulted from training (first number in the output file of word2vec):\n");
    printf("\t\t-vocab_size <int>\n");
    printf("\tValue for the layer size used in training (second number in the output file of word2vec):\n");
    printf("\t\t-layer_size <int>\n");
    printf("\tValue for the window size:\n");
    printf("\t\t-window <int>\n\n");
    return 0;
  } //reading command line arguments
  if ((i = ArgPos((char *)"-layer_size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-vocab_size", argc, argv)) > 0) vocab_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);  

  // allocating memory to store the network elements
  syn0 = (real *)calloc(layer1_size*vocab_size,sizeof(real)); 
  syn1 = (real *)calloc(layer1_size*vocab_size,sizeof(real)); 
  neu1 = (real *)calloc(layer1_size,sizeof(real));

  index_buff = (char *)calloc(MAX_INDEX_BUFF_SIZE,sizeof(char));
  // reading the network from file
  read_syn0();
  read_syn1();

  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real)); //allocating memory for expTable
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table in the same way as in word2vec
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  //building the vocabulary and the vocabulary hash from the files it was stored in
  BuildVocabFromFile();
  BuildVocabHashFromFile();

  int length = 0; //word lenght of sentence variable
  int syno_length = 0; //how many synonyms/replacements
  long long * sen; //sentence variable where words are represented as vocabualry indices
  long long * sen_temp; //temporary sentence variable where words are represented as vocabulary indices
  sen_temp = (long long *)calloc(MAX_SENTENCE_LENGTH,sizeof(long long)); //allocating memory for sen_temp
  long long * synonym; //replacement word (in vocabulary index form)
  long double prob = 0; //probability variable 
  long long ptr = 0, ptr_temp = 0; //pointer used to go through the sentences file
  long long syno_ptr = 0, syno_ptr_temp = 0; //pointer used to go through the synonyms/replacements file 


  FILE *sentfile = fopen("sentences","r");
  FILE *indices = fopen("indices","r");
  FILE *synfile = fopen("synonyms","r");
  FILE *fo = fopen("wordprobs","w");
  int lines = 0;
  char line[MAX_SENTENCE_LENGTH]; // buffer to store current sentence
  char synline[MAX_SENTENCE_LENGTH]; // buffer to store synonyms


  lines = Lines(sentfile); // how many lines in the sentences file, which is used as the outer loop delimiter 
                     //(this can be done) since all the files "sentences", "synonyms" and "indices" have the same number of lines delimited by "\n"
  rewind(sentfile);
  rewind(synfile);
  
  for(i = 0; i<lines; i++){  //outer loop iterating through "sentences", "synonyms" and "indices" line by line

    // read sentence
    ptr = ftell(sentfile); // store beginning of line
    if (readLine(sentfile,line) < 0) break;
    length = LineWordCount(line); 
    //printf("sent words %d\n",length);

    // read word replacements
    syno_ptr = ftell(synfile); // store beginning of line
    if (readLine(synfile,synline) < 0) break;
    syno_length = LineWordCount(synline);
    printf("synline %s\n",synline);

    fseek(sentfile,ptr,SEEK_SET); // move the pointer back to the beginning of the line
    sen = FileToSen(length,sentfile); //sen is an array of longs with the words of the sentence in a vocabulary index format

    fseek(synfile,syno_ptr,SEEK_SET);
    synonym = FileToSen(syno_length,synfile); //synonym is an array of longs with the replacements/synonyms from the "synonyms" file in vocabulary index format

    fseek(sentfile,1,SEEK_CUR); // added to get past newline
    fseek(synfile,1,SEEK_CUR);
 
    ReadIndexFromFile(indices); //reads the index and puts it in the char array "index_buff"
    target_index = GetIndex(); //returns a numerical value from what is in the char array "index_buff"
    for(k=0; k<syno_length;k++){ //repeats forward propagation for each synonym in the line
      memcpy(sen_temp,sen,MAX_SENTENCE_LENGTH*sizeof(long long)); //copying the sentence into sen_temp where synonyms will be changed
      sen_temp[target_index] = synonym[k]; //replacing the target word with a synonym/replacement
      prob = ForwardPropagate(length,sen_temp); //doing forward propagation to get the probability
      //prob = prob * 100000; // multiplying the probabilty by 100000 or taking the negative log is done in this line

      fprintf(fo,"%s %Lf\n",vocab[synonym[k]].word,prob); // SEA the replacement word and its probability 
    }
  }

  fclose(fo);
  fclose(sentfile);
  fclose(synfile);
  fclose(indices);

  return 0;
}
