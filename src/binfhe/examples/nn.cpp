// ----------------------------------------------------------------------------|
// Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
// Project    : Demonstrate Fast Fully Homomorphic Evaluation of Encrypted Inputs
//              using Deep Discretized Neural Networks hence preserving Privacy
// ----------------------------------------------------------------------------|
// File       : nn.cpp
// Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
//              Michele Minelli     <Michele.Minelli@ens.fr>
//              Matthias Minihold   <Matthias.Minihold@RUB.de>
//              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
// Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
//              https://github.com/tfhe
// ----------------------------------------------------------------------------|
// Description:
//     Showcases how to efficiently evaluate privacy-perserving neural networks.
// ----------------------------------------------------------------------------|
// Revisions  :
// Date        Version  Description
// 2017-11-16  0.3.0    Version for github, referenced by ePrint paper
// ----------------------------------------------------------------------------|


// Includes
#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include "binfhe-constants.h"
#include "lattice/lat-hal.h"
#include <sys/time.h>
// Multi-processing
#include <sys/wait.h>
#include <unistd.h>

//! binfhecontext
#define PROFILE
#include <vector>
#include "binfhecontext.h"
using namespace lbcrypto;


// Defines
#define VERBOSE 1
#define STATISTICS true
#define WRITELATEX false
#define N_PROC 4

// Security constants
#define SECLEVEL 80
#define SECNOISE true
#define SECALPHA pow(2., -20)
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_n  600                   ///  LweParams
#define SEC_PARAMS_N 1024                   /// TLweParams
#define SEC_PARAMS_k    1                   /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -36)   /// TLweParams
#define SEC_PARAMS_BK_BASEBITS 10           /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
#define SEC_PARAMS_KS_STDDEV pow(2., -25)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH   18           /// Key Switching Params

// The expected topology of the provided neural network is 256:30:10
#define NUM_NEURONS_LAYERS 3
#define NUM_NEURONS_INPUT  256
#define NUM_NEURONS_HIDDEN 30
#define NUM_NEURONS_OUTPUT 10

#define CARD_TESTSET 10000

// Files are expected in the executable's directory
#define PATH_TO_FILES       "buildotests/test/" //TODO FIXME!
#define FILE_TXT_IMG        "/home/azency/Code/Work/dinn_binfhe/weights-and-biases/txt_img_test.txt"
#define FILE_TXT_BIASES     "/home/azency/Code/Work/dinn_binfhe/weights-and-biases/txt_biases.txt"
#define FILE_TXT_WEIGHTS    "/home/azency/Code/Work/dinn_binfhe/weights-and-biases/txt_weights.txt"
#define FILE_TXT_LABELS     "/home/azency/Code/Work/dinn_binfhe/weights-and-biases/txt_labels.txt"
#define FILE_LATEX          "results_LaTeX.tex"
#define FILE_STATISTICS     "results_stats.txt"

// Tweak neural network
#define THRESHOLD_WEIGHTS  9
#define THRESHOLD_SCORE -100

#define MSG_SLOTS    700
#define TORUS_SLOTS  400


using namespace std;

void deleteTensor(int*** tensor, int dim_mat, const int* dim_vec);
void deleteMatrix(int**  matrix, int dim_mat);




int main(int argc, char **argv)
{
    // Security
    const int minimum_lambda = SECLEVEL;
    const bool noisyLWE      = SECNOISE;
    const double alpha       = SECALPHA;

    // Input data
    const int n_images = CARD_TESTSET;

    // Network specific
    const int num_wire_layers = NUM_NEURONS_LAYERS - 1;
    const int num_neuron_layers = NUM_NEURONS_LAYERS;
    const int num_neurons_in = NUM_NEURONS_INPUT;
    const int num_neurons_hidden = NUM_NEURONS_HIDDEN;
    const int num_neurons_out = NUM_NEURONS_OUTPUT;

    // Vector of number of neurons in layer_in, layer_H1, layer_H2, ..., layer_Hd, layer_out;
    const int topology[num_neuron_layers] = {num_neurons_in, num_neurons_hidden, num_neurons_out};

    const int space_msg = MSG_SLOTS;
    const int space_after_bs = TORUS_SLOTS;

    const bool clamp_biases  = false;
    const bool clamp_weights = false;

    const bool statistics        = STATISTICS;
    const bool writeLaTeX_result = WRITELATEX;

    const int threshold_biases  = THRESHOLD_WEIGHTS;
    const int threshold_weights = THRESHOLD_WEIGHTS;
    const int threshold_scores  = THRESHOLD_SCORE;

    // Program the wheel to value(s) after Bootstrapping
    // const Torus32 mu_boot = modSwitchToTorus32(1, space_after_bs);

    const int total_num_hidden_neurons = n_images * NUM_NEURONS_HIDDEN;  //TODO (sum all num_neurons_hidden)*n_images
    const double avg_bs  = 1./NUM_NEURONS_HIDDEN;
    const double avg_total_bs  = 1./total_num_hidden_neurons;
    const double avg_img = 1./n_images;
    const double clocks2seconds = 1. / CLOCKS_PER_SEC;
    const int slice = (n_images+N_PROC-1)/N_PROC;

    // Huge arrays
    int*** weights = new int**[num_wire_layers];  // allocate and fill matrices holding the weights
    NativeInteger*** weights_1 = new NativeInteger**[num_wire_layers];
    int ** biases  = new int* [num_wire_layers];  // allocate and fill vectors holding the biases
    int ** images  = new int* [n_images];
    int  * labels  = new int  [n_images];

    // Temporary variables
    string line;
    int el, l;
    int num_neurons_current_layer_in, num_neurons_current_layer_out;


    if (VERBOSE)
    {
        cout << "Starting experiment to classify " << n_images;
        if (!noisyLWE) cout << " noiseless";
        cout << " encrypted MNIST images." << endl;
        cout << "(Run: " << argv[0] << " )" << endl;
        cout << "Execution with parameters... alpha = " << alpha << ", number of processes: " << N_PROC << endl;
    }



    //! binfhecontext初始化 strat
    auto cc = BinFHEContext();

    int p = 8; //明文模长，要把[0,p]迁移到[-p/2,p/2]上
    // int p      = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  //! 这里的最大明文空间有问题，返回的是q/256,不知道为啥。。。。。。

    // uint32_t n = 128;
    // uint32_t N = 1<<12;
    // NativeInteger q(1<<13); //q|2N
    // NativeInteger Q(((1<<20) + 1));
    // // double std = pow(2,-20);
    // double std = 0.01;
    // uint32_t baseKS = 1<<1;
    // uint32_t baseG = 1<<1;
    // uint32_t baseR = 2;
    
    // cc.GenerateBinFHEContext(n, N, q, Q, std, baseKS, baseG, baseR, GINX);

    cc.GenerateBinFHEContext();
    int q = cc.GetParams()->GetLWEParams()->Getq().ConvertToInt();


    // Sample Program: Step 2: Key Generation
    // Generate the secret key
    auto sk = cc.KeyGen();



    // cout<<"密钥是"<<sk->GetElement()<<endl;

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;



    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1/2)
            return NativeInteger(1);
        else
            return p1 - 1 ;
    };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);

    
    

    //! binfhecontext初始化 end


    if (VERBOSE) cout << "IMPORT PIXELS, WEIGHTS, BIASES, and LABELS FROM FILES" << endl;
    if (VERBOSE) cout << "Reading images (regardless of dimension) from " << FILE_TXT_IMG << endl;
    ifstream file_images(FILE_TXT_IMG);

    for (int img=0; img<n_images; ++img)
        images[img] = new int[num_neurons_in];

    int filling_image = 0;
    int image_count = 0;
    while(getline(file_images, line))
    {
        images[filling_image][image_count++] = stoi(line);
        if (image_count == num_neurons_in)
        {
            image_count = 0;
            filling_image++;
        }
    }
    file_images.close();


    if (VERBOSE) cout << "Reading weights from " << FILE_TXT_WEIGHTS << endl;
    ifstream file_weights(FILE_TXT_WEIGHTS);

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        weights[l] = new int*[num_neurons_current_layer_in];
        weights_1[l] = new NativeInteger*[num_neurons_current_layer_in];
        for (int i = 0; i<num_neurons_current_layer_in; ++i)
        {
            weights[l][i] = new int[num_neurons_current_layer_out];
            weights_1[l][i] = new NativeInteger[num_neurons_current_layer_out];
            for (int j=0; j<num_neurons_current_layer_out; ++j)
            {
                getline(file_weights, line);
                el = stoi(line);
                if (clamp_weights)
                {
                    if (el < -threshold_weights)
                        el = -threshold_weights;
                    else if (el > threshold_weights)
                        el = threshold_weights;
                    // else, nothing as it holds that: -threshold_weights < el < threshold_weights
                }
                weights[l][i][j] = el;
                // NativeInteger a = NativeInteger((el+p)%p);
                weights_1[l][i][j] = NativeInteger((el+q)%q);
            }
        }
    }
    file_weights.close();


    if (VERBOSE) cout << "Reading biases from " << FILE_TXT_BIASES << endl;
    ifstream file_biases(FILE_TXT_BIASES);

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        biases [l] = new int [num_neurons_current_layer_out];
        for (int j=0; j<num_neurons_current_layer_out; ++j)
        {
            getline(file_biases, line);
            el = stoi(line);
            if (clamp_biases)
            {
                if (el < -threshold_biases)
                    el = -threshold_biases;
                else if (el > threshold_biases)
                    el = threshold_biases;
                // else, nothing as it holds that: -threshold_biases < el < threshold_biases
            }
            biases[l][j] = el;
        }
    }
    file_biases.close();


    if (VERBOSE) cout << "Reading labels from " << FILE_TXT_LABELS << endl;
    ifstream file_labels(FILE_TXT_LABELS);
    for (int img=0; img<n_images; ++img)
    {
        getline(file_labels, line);
        labels[img] = stoi(line);
    }
    file_labels.close();

    if (VERBOSE) cout << "Import done. END OF IMPORT" << endl;



    // Temporary variables and Pointers to existing arrays for convenience
    bool notSameSign;
    // Torus32 mu, phase;

    int** weight_layer;
    int * bias;
    int * image;
    int pixel, label;
    int x, w, w0;

    //! 后续要用到的便量，提前定义；
    NativeInteger w_1;

    // LweSample *multi_sum, *enc_image, *bootstrapped;

    

    //! 图片的密文，以vector的型式存储
    vector<LWECiphertext> enc_imgae_1;

    //! multi_sum，中间变量，计算每一层的值
    vector<LWECiphertext> multi_sum_1;

    //! bootstrapped_1
    vector<LWECiphertext> bootstrapped_1;



    int multi_sum_clear[num_neurons_hidden];
    int output_clear   [num_neurons_out];

    int max_score = 0;
    int max_score_clear = 0;
    int class_enc = 0;
    int class_clear = 0;
    int score = 0;
    LWEPlaintext score_1;
    int score_clear = 0;


    bool failed_bs = false;
    // Counters
    int count_errors = 0;
    int count_errors_with_failed_bs = 0;
    int count_disagreements = 0;
    int count_disagreements_with_failed_bs = 0;
    int count_disag_pro_clear = 0;
    int count_disag_pro_hom = 0;
    int count_wrong_bs = 0;

    int r_count_errors, r_count_disagreements, r_count_disag_pro_clear, r_count_disag_pro_hom, r_count_wrong_bs, r_count_errors_with_failed_bs, r_count_disagreements_with_failed_bs;
    double r_total_time_network, r_total_time_bootstrappings;

    // For statistics output
    double avg_time_per_classification = 0.0;
    double avg_time_per_bootstrapping = 0.0;
    double total_time_bootstrappings = 0.0;
    double total_time_network = 0.0;
    double error_rel_percent = 0.0;

    // Timings
    clock_t bs_begin, bs_end, net_begin, net_end;
    double time_per_classification, time_per_bootstrapping, time_bootstrappings;


    // for (int img = 0; img < ( (+1)*slice) && (img< n_images); /*img*/ )
    for (int img = 0;(img< n_images); /*img*/ )
            {
                image = images[img];
                label = labels[img];
                ++img;

                // Generate encrypted inputs for NN (LWE samples for each image's pixels on the fly)
                // To be generic...
                num_neurons_current_layer_out= topology[0];
                num_neurons_current_layer_in = num_neurons_current_layer_out;

                // enc_image = new_LweSample_array(num_neurons_current_layer_in, in_out_params);

                for (int i = 0; i < num_neurons_current_layer_in; ++i)
                {
                    pixel = image[i];
                    //! change to message space 
                    // mu = modSwitchToTorus32(pixel, space_msg);
                    if (noisyLWE)
                    {
                        //! 进行加密
                        // lweSymEncrypt(enc_image + i, mu, alpha, secret->lwe_key);
                        //! 这里放进来，delte的时候清空它
                        auto ct = cc.Encrypt(sk, (pixel+p) % p, FRESH, p);
                        enc_imgae_1.push_back(ct);

                    }
                    else
                    {
                        // lweNoiselessTrivial(enc_image + i, mu, in_out_params);
                    }
                }

                // ========  FIRST LAYER(S)  ========
                net_begin = clock();

                // multi_sum = new_LweSample_array(num_neurons_current_layer_out, in_out_params);
                for (l=0; l<num_wire_layers - 1 ; ++l)     // Note: num_wire_layers - 1 iterations; last one is special. Access weights from level l to l+1.
                {
                    // To be generic...
                    num_neurons_current_layer_in = num_neurons_current_layer_out;
                    num_neurons_current_layer_out= topology[l+1];
                    bias = biases[l];
                    weight_layer = weights[l];
                    for (int j=0; j<num_neurons_current_layer_out; ++j)
                    {
                        w0 = bias[j];
                        multi_sum_clear[j] = w0;
                        //! change to message space
                        auto ct = cc.EvalConstant((w0+p) % p, p);
                        multi_sum_1.push_back(ct);
                        // LWEPlaintext temp;
                        // cc.Decrypt(sk, ct, &temp, p);

                        // cout<<"bais"<<j<<"结果比较clear和enc      "<<multi_sum_clear[j]<<"      "<<temp<<endl;

                        for (int i=0; i<num_neurons_current_layer_in; ++i)
                        {
                            x = image [i];  // clear input
                            // int64_t temp_x;
                            // cc.Decrypt(sk, enc_imgae_1[i], &temp_x, p);
                            // cout<<"像素是"<<x<<"       "<<temp_x<<endl;

                            w = weight_layer[i][j];  // w^dagger
                            multi_sum_clear[j] += x * w; // process clear input

                            //! 直接读取w_1
                            w_1 = weights_1[l][i][j];
                            //! 准备相乘,权重乘以像素,每次都要重新构造，非常减低效率，怎样去避免；
                            auto temp_A = enc_imgae_1[i]->GetA().ModMul(w_1);
                            auto temp_B = enc_imgae_1[i]->GetB().ModMul(w_1, enc_imgae_1[i]->GetModulus());

                            //! 加到multi_sum上
                            multi_sum_1[j]->GetA().ModAddEq(temp_A);
                            multi_sum_1[j]->GetB().ModAddEq(temp_B, multi_sum_1[j]->GetModulus());

                            // cout<<"密文的模场"<<multi_sum_1[j]->GetModulus()<<endl;

                            // int64_t temp;
                            // cc.Decrypt(sk, multi_sum_1[j], &temp, p);
                            // temp = (temp<p/2)? temp:temp-p;
                            // if(temp != multi_sum_clear[j] ){
                            //     cout<<j<<"个神经云"<<i<<"次运算"<<"出错结果比较clear和enc      "<<multi_sum_clear[j]<<"      "<<temp<<endl;
                            // }
                            
                        }
                    }
                }

                // Bootstrap multi_sum
                // bootstrapped = new_LweSample_array(num_neurons_current_layer_out, in_out_params);
                bs_begin = clock();
                for (int j=0; j<num_neurons_current_layer_out; ++j)
                {
                    /**
                     *  Bootstrapping results in each coordinate 'bootstrapped[j]' to contain an LweSample
                     *  of low-noise (= fresh LweEncryption) of 'mu_boot*phase(multi_sum[j])' (= per output neuron).
                     */
                    //! bootstrapping的同时，做一次示性函数，result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
                    auto ct_sign = cc.EvalFunc(multi_sum_1[j], lut);

                    // auto ct_sign == cc.EvalSign(multi_sum_1[1]);

                    // auto ct_sign =cc.GetBinFHEScheme()->MyEvalFunc(cc.GetParams(), cc.GetBTKey() , multi_sum_1[j], p);
                    
                    // auto ct_sign = cc.EvalSign(multi_sum_1[j]);
                    // cout<<"自举后的LWE模长度"<<ct_sign->GetA().GetModulus()<<endl;
                    LWEPlaintext temp,temp1;
                    // cc.Decrypt(cc.GetBTKey().skN, ct_sign, &temp, p);
                    cc.Decrypt(sk, ct_sign, &temp, p);
                    cc.Decrypt(sk, multi_sum_1[j], &temp1, p);
                    temp = (temp<p/2)? temp:temp-p;
                    temp1 = (temp1<p/2)? temp1:temp1-p;
                    if (temp*multi_sum_clear[j] <= 0 || true){
                        cout<<"自举前的真实值和、密文值 和 自举后的密文值"<<multi_sum_clear[j]<< "     " <<temp1<< "     "<<temp<<endl;
                    } 
                    
                    bootstrapped_1.push_back(ct_sign);

                }

                bs_end = clock();
                time_bootstrappings = bs_end - bs_begin;
                // cout<< time_bootstrappings;
                total_time_bootstrappings += time_bootstrappings;
                time_per_bootstrapping = time_bootstrappings*avg_bs;
                if (VERBOSE) cout <<  time_per_bootstrapping*clocks2seconds << " [sec/bootstrapping]" << endl;

                //! 清空multi_sum_1
                multi_sum_1.clear();


                // ========  LAST (SECOND) LAYER  ========
                max_score = threshold_scores;
                max_score_clear = threshold_scores;

                bias = biases[l];
                weight_layer = weights[l];
                l++;
                num_neurons_current_layer_in = num_neurons_current_layer_out;
                num_neurons_current_layer_out= topology[l]; // l == L = 2
                // multi_sum = new_LweSample_array(num_neurons_current_layer_out, in_out_params); // TODO possibly overwrite storage
                for (int j=0; j<num_neurons_current_layer_out; ++j)
                {
                    w0 = bias[j];
                    output_clear[j] = w0;
                    //! 同样地，把bias先加上去
                    auto ct = cc.EvalConstant((w0+p)%p, p);
                    multi_sum_1.push_back(ct);
                    LWEPlaintext temp;
                    cc.Decrypt(sk, multi_sum_1[j], &temp, p);

                    cout<<"第二层 bais 结果比较clear和enc      "<<output_clear[j]<<"      "<<temp<<endl;                    

                    for (int i=0; i<num_neurons_current_layer_in; ++i)
                    {
                        w = weight_layer[i][j];
                        //! 乘起来加进去
                        // lweAddMulTo(multi_sum + j, w, bootstrapped + i, in_out_params); // process encrypted input

                        // //! 将w转换到NativeInteger上，也是每次才进行构造，效率太低了
                        // NativeInteger* w_1 = new NativeInteger;
                        // w_1->SetValue(to_string((w+5*plain_mod)%plain_mod));

                        //! 直接读取w1
                        w_1 = weights_1[l-1][i][j]; 
                        //! 准备相乘,权重乘以像素,每次都要重新构造，非常减低效率，怎样去避免；
                        auto temp_A = bootstrapped_1[i]->GetA().ModMul(w_1);
                        auto temp_B = bootstrapped_1[i]->GetB().ModMulFast(w_1, bootstrapped_1[i]->GetModulus());
                        // (*temp)->GetB().ModMulFastEq(w_1,cc.GetMaxPlaintextSpace());

                        //! 加到multi_sum上
                        multi_sum_1[j]->GetA().ModAddEq(temp_A);
                        multi_sum_1[j]->GetB().ModAddFastEq(temp_B, multi_sum_1[j]->GetModulus());

                        


                        // process clear input
                        if (multi_sum_clear[i] < 0)
                            output_clear[j] -= w;
                        else
                            output_clear[j] += w;

                        LWEPlaintext temp;
                        cc.Decrypt(sk, multi_sum_1[j], &temp, p);
                        cout<<"第二层 bais 结果比较clear和enc      "<<output_clear[j]<<"      "<<temp<<endl; 

                        

                    }
                    //!这里进行解密
                    // score = lwePhase(multi_sum + j, secret->lwe_key);

                    
                    cc.Decrypt(sk, multi_sum_1[j], &score_1, p);
                    score_1 = (score_1>p/2)? score_1%p-p: score_1%p;
                    if (score_1 > max_score)
                    {
                        max_score = score_1;
                        class_enc = j;
                    }
                    score_clear = output_clear[j];
                    if (score_clear > max_score_clear)
                    {
                        max_score_clear = score_clear;
                        class_clear = j;
                    }

                    //! 比较
                    cout<<"score_1 和 score_clear 分别是"<<score_1<<"   "<<score_clear<<endl;
                }

                if (class_enc != label)
                {
                    count_errors++;
                    if (failed_bs)
                        count_errors_with_failed_bs++;
                }

                if (class_clear != class_enc)
                {
                    count_disagreements++;
                    if (failed_bs)
                        count_disagreements_with_failed_bs++;

                    if (class_clear == label)
                        count_disag_pro_clear++;
                    else if (class_enc == label)
                        count_disag_pro_hom++;
                }
                net_end = clock();
                time_per_classification = net_end - net_begin;
                total_time_network += time_per_classification;
                if (VERBOSE) cout << "            "<< time_per_classification*clocks2seconds <<" [sec/classification]" << endl;
                
                // free memory
                // delete_LweSample_array(num_neurons_in,     enc_image);
                // delete_LweSample_array(num_neurons_hidden, bootstrapped);
                // delete_LweSample_array(num_neurons_out,    multi_sum);
                enc_imgae_1.clear();
                bootstrapped_1.clear();
                multi_sum_1.clear();



    }


    if (statistics)
    {
        ofstream of(FILE_STATISTICS);
        // Print some statistics
        error_rel_percent = count_errors*avg_img*100;
        avg_time_per_classification = time_per_classification*avg_img*clocks2seconds;
        avg_time_per_bootstrapping  = time_per_bootstrapping *avg_total_bs *clocks2seconds;

        cout << "Errors: " << count_errors << " / " << n_images << " (" << error_rel_percent << " %)" << endl;
        cout << "Disagreements: " << count_disagreements;
        cout << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        cout << "Wrong bootstrappings: " << count_wrong_bs << endl;
        cout << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        cout << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        cout << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        cout << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;

        of << "Errors: " << count_errors << " / " << n_images << " (" << error_rel_percent << " %)" << endl;
        of << "Disagreements: " << count_disagreements;
        of << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        of << "Wrong bootstrappings: " << count_wrong_bs << endl;
        of << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        of << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        of << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        of << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;

        // Write some statistics
        cout << "\n Wrote statistics to file: " << FILE_STATISTICS << endl << endl;
        of.close();
    }

    if (writeLaTeX_result)
    {
        cout << "\n Wrote LaTeX_result to file: " << FILE_LATEX << endl << endl;
        ofstream of(FILE_LATEX);
        of << "%\\input{"<<FILE_LATEX<<"}" << endl;

        of << "% Experiments detailed" << endl;
        of << "\\newcommand{\\EXPnumBS}{$"<<total_num_hidden_neurons<<"$}" << endl;
        of << "\\newcommand{\\EXPbsEXACT}{$"    <<avg_time_per_bootstrapping<<"$\\ [sec/bootstrapping]}" << endl;
        of << "\\newcommand{\\EXPtimeEXACT}{$"  <<avg_time_per_classification<<"$\\ [sec/classification]}" << endl;

        of << "\\newcommand{\\EXPnumERRabs}{$"  <<count_errors<<"$}" << endl;
        of << "\\newcommand{\\EXPnumERRper}{$"  <<error_rel_percent<<"\\ \\%$}" << endl;
        of << "\\newcommand{\\EXPwrongBSabs}{$" <<count_wrong_bs<<"$}" << endl;
        of << "\\newcommand{\\EXPwrongDISabs}{$"<<count_disagreements_with_failed_bs<<"$}" << endl;
        of << "\\newcommand{\\EXPdis}{$"        <<count_disagreements<<"$}" << endl;
        of << "\\newcommand{\\EXPclear}{$"      <<count_disag_pro_clear<<"$}" << endl;
        of << "\\newcommand{\\EXPhom}{$"        <<count_disag_pro_hom<<"$}" << endl << endl;

        of << "\\begin{Verbatim}[frame=single,numbers=left,commandchars=+\\[\\]%" << endl;
        of << "]" << endl;
        of << "### Classified samples: +EXPtestset" << endl;
        of << "Time per bootstrapping: +EXPbsEXACT" << endl;
        of << "Errors: +EXPnumERRabs / +EXPtestset (+EXPnumERRper)" << endl;
        of << "Disagreements: +EXPdis" << endl;
        of << "(pro-clear/pro-hom: +EXPclear / +EXPhom)" << endl;
        of << "Wrong bootstrappings: +EXPwrongBSabs" << endl;
        of << "Disagreements with wrong bootstrapping: +EXPwrongDISabs" << endl;
        of << "Avg. time for the evaluation of the network: +EXPtimeEXACT" << endl;
        of << "\\end{Verbatim}" << endl;
        of.close();
    }

    // free memory
    // delete_gate_bootstrapping_secret_keyset(secret);
    // delete_gate_bootstrapping_parameters(params);

    deleteTensor(weights,num_wire_layers, topology);
    deleteMatrix(biases, num_wire_layers);
    deleteMatrix(images, n_images);
    delete[] labels;

    return 0;
}


void deleteTensor(int*** tensor, int dim_tensor, const int* dim_vec)
{
    int** matrix;
    int dim_mat;
    for (int i=0; i<dim_tensor; ++i)
    {
        matrix =  tensor[i];
        dim_mat = dim_vec[i];
        deleteMatrix(matrix, dim_mat);
    }
    delete[] tensor;
}


void deleteMatrix(int** matrix, int dim_mat)
{
    for (int i=0; i<dim_mat; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}


