#include <stdio.h>
#include<fstream>
#include <vector>
#include <string>
#include <map>
#include<iostream>
using namespace std;
double calculateError(float x, float y, float ex, float ey,float cov )
{
  return (x/y)*(sqrt(((ex/x)*(ex/x))+((ey/y)*(ey/y))-(2*cov/x*y)));
    }
void OverlayPlots(string filename, string filename1, string filename2)
{
  char* hname = new char[200];
  char* hname1 = new char[200];
  char* hname2 = new char[200];

  char* hist_name  = new char[200];
  char* hist_name1 = new char[200];
  char* hist_name2 = new char[200];
  char* hist_name3  = new char[200];
  char* hist_name4 = new char[200];
  char* hist_name5 = new char[200];
char* hist_name6  = new char[200];
  char* hist_name7 = new char[200];
  char* hist_name8 = new char[200];
 char* hist_name9 = new char[200];
  char* full_path = new char[2000];
  char* full_path1 = new char[2000];
  char* full_path2 = new char[2000];
  char* path2 = new char[2000];
  char* title= new char[2000];
  char* title1= new char[2000];char* title2= new char[2000];
  char* full_path3 = new char[2000];
  char* full_path4 = new char[2000];
  char* full_path5 = new char[2000];
  char* full_path6 = new char[2000];
  char* full_path7 = new char[2000];
  char* full_path8 = new char[2000];
  char* full_path9 = new char[2000];
  char* full_path10 = new char[2000];
  char* full_path11= new char[2000];
  sprintf(hname,"%s",filename.c_str());
  sprintf(hname1,"%s",filename1.c_str());
  sprintf(hname2,"%s",filename2.c_str());

  sprintf(path2,"Results");//,filename.c_str());
  TFile * inputfile = new TFile(hname,"READ");
  TFile * inputfile1 = new TFile(hname1,"READ");
   TFile * inputfile2 = new TFile(hname2,"READ");

  char* reso_name=new char[1000];
  char* resp_name=new char[1000];
  char* name=new char[1000];
  char* name1=new char[1000];
  char* name2=new char[1000];
const char *data[4] = { "Valid","Tbdata" , "QGSP","FTFP"};

 const char *eve_cat[2] = { "SSinEE", "MipsInEE"};	   

 
// for(int i_data=0; i_data<2; i_data++)
//   {
 
 sprintf(hist_name,"Response_Valid");//, data[i_data]);
 sprintf(hist_name1,"Resolution_Valid");//,data[i_data]);
 sprintf(hist_name2,"Response_Tbdata");
 sprintf(hist_name3,"Resolution_Tbdata");

 
     TGraphErrors* h_resolution = (TGraphErrors*)inputfile->Get(hist_name1);
    TGraphErrors* h_response = (TGraphErrors*)inputfile->Get(hist_name);
    TGraphErrors* h_resolution1 = (TGraphErrors*)inputfile->Get(hist_name3);
    TGraphErrors* h_response1 = (TGraphErrors*)inputfile->Get(hist_name2);

    sprintf(hist_name4,"Response_SSinEE_Tbdata");
    sprintf(hist_name5,"Resolution_SSinEE_Tbdata");
    sprintf(hist_name6,"Response_MipsInEE_Tbdata");
    sprintf(hist_name7,"Resolution_MipsInEE_Tbdata");
    sprintf(hist_name8,"response_data_chi2methodEbeam_all_gaus_data.txt");
    sprintf(hist_name9,"resolution_data_chi2methodEbeam_all_gaus_data.txt");

    TGraphErrors* h_resolution_EE = (TGraphErrors*)inputfile1->Get(hist_name5);
    TGraphErrors* h_response_EE = (TGraphErrors*)inputfile1->Get(hist_name4);
    TGraphErrors* h_resolution_FH = (TGraphErrors*)inputfile1->Get(hist_name7);
    TGraphErrors* h_response_FH = (TGraphErrors*)inputfile1->Get(hist_name6);
    TGraphErrors* h_resolution_shubham = (TGraphErrors*)inputfile2->Get(hist_name9);
    TGraphErrors* h_response_shubham = (TGraphErrors*)inputfile2->Get(hist_name8);

    sprintf(full_path1,"%s/overlay_resolution_v1_Evnt_v1.png",path2);//,path2,data[i_data]);
    sprintf(full_path2,"%s/overlay_response_v1_Evnt_v1.png",path2);//,path2,data[i_data]);
    TMultiGraph* mg = new TMultiGraph();
    TMultiGraph* mg1 = new TMultiGraph();
// auto legend = new TLegend(0.7,0.6,0.9,0.9);
//   legend->SetHeader("","C");
//  gStyle->SetLegendTextSize(0.04);
  // auto legend1 = new TLegend(0.7,0.1,0.9,0.4);
  // legend1->SetHeader("","C");
  // mg->Add(h_resolution);
  // mg->Add(h_resolution1);
  // mg->Add(h_resolution_EE);
  // mg->Add(h_resolution_FH);
  // mg1->Add(h_response);
  // mg1->Add(h_response1);
  // mg1->Add(h_response_EE);
  // mg1->Add(h_response_FH);
  // mg->SetTitle("resolution for pions  ");
  // mg->GetXaxis()->SetTitle("Beam energy (GeV)");
  // mg->GetYaxis()->SetTitleOffset(1.4);
  // mg->GetYaxis()->SetTitle("sigma/mean");

  //  mg1->SetTitle("response for pions  ");
  // mg1->GetXaxis()->SetTitle("Beam energy (GeV)");
  // mg1->GetYaxis()->SetTitleOffset(1.4);
  // mg1->GetYaxis()->SetTitle("mean/beam energy");

  //  h_resolution->SetMarkerStyle(8);
   
    // h_resolution->SetTitle(" ");
    // h_resolution->GetXaxis()->SetTitle("Beam energy (GeV)");
    // h_resolution->GetYaxis()->SetTitleOffset(1.4);
    // h_resolution->GetYaxis()->SetTitle("sigma/Mean (GeV)");
    h_resolution->SetMarkerColorAlpha(kRed, 0.95);
    h_resolution->SetMarkerSize(2);
    h_resolution->SetMarkerStyle(8);
    
    h_resolution1->SetMarkerColorAlpha(kBlack, 0.95);
    h_resolution1->SetMarkerSize(2);
    h_resolution1->SetMarkerStyle(21);
    h_resolution_EE->SetMarkerColorAlpha(6, 0.95);
    h_resolution_EE->SetMarkerSize(3);
    h_resolution_EE->SetMarkerStyle(kFullDiamond);
    
    h_resolution_FH->SetMarkerColorAlpha(kBlue-1, 0.95);
    h_resolution_FH->SetMarkerSize(3);
    h_resolution_FH->SetMarkerStyle(23);
    h_resolution_shubham->SetMarkerColorAlpha(kBlue, 0.95);
    h_resolution_shubham->SetMarkerSize(3);
    h_resolution_shubham->SetMarkerStyle(32);

    h_response_shubham->SetMarkerColorAlpha(kBlue, 0.95);
    h_response_shubham->SetMarkerSize(3);
    h_response_shubham->SetMarkerStyle(32);

    h_resolution_shubham->SetLineColor(kBlue);
    h_resolution_shubham->SetLineStyle(2);
    h_resolution_shubham->SetLineWidth(2);
 h_response_shubham->SetLineColor(kBlue);
    h_response_shubham->SetLineStyle(2);
    h_response_shubham->SetLineWidth(2);

    h_response->SetMarkerColorAlpha(kRed, 0.95);
    h_response->SetMarkerSize(2);
    h_response->SetMarkerStyle(8);

    h_response1->SetMarkerColorAlpha(kBlack, 0.95);
    h_response1->SetMarkerSize(2);
    h_response1->SetMarkerStyle(21);
    h_response_EE->SetMarkerColorAlpha(6, 0.95);
    h_response_EE->SetMarkerSize(3);
    h_response_EE->SetMarkerStyle(kFullDiamond);

    h_response_FH->SetMarkerColorAlpha(kBlue-1, 0.95);
    h_response_FH->SetMarkerSize(3);
    h_response_FH->SetMarkerStyle(23);
 mg->Add(h_resolution);
  mg->Add(h_resolution1);
  mg->Add(h_resolution_EE);
  mg->Add(h_resolution_FH);
  mg->Add(h_resolution_shubham);
  mg1->Add(h_response_shubham);
  mg1->Add(h_response);
  mg1->Add(h_response1);
  mg1->Add(h_response_EE);
  mg1->Add(h_response_FH);

  // mg->SetTitle("resolution for pions  ");
  mg->GetXaxis()->SetTitle("Beam energy (GeV)");
  mg->GetYaxis()->SetTitleOffset(1.4);
  mg->GetYaxis()->SetTitle("sigma/mean");

  //  mg1->SetTitle("response for pions  ");
  mg1->GetXaxis()->SetTitle("Beam energy (GeV)");
  mg1->GetYaxis()->SetTitleOffset(1.4);
  mg1->GetYaxis()->SetTitle("mean/beam energy");


    h_resolution1->SetTitle(" ");
 //    h_resolution1->GetXaxis()->SetTitle("Beam energy (GeV)");
 //    h_resolution1->GetYaxis()->SetTitleOffset(1.4);
 //    h_resolution1->GetYaxis()->SetTitle("sigma/Mean (GeV)");
 //    h_resolution1->SetMarkerColorAlpha(kGreen+3, 0.95);
 //    h_resolution1->SetMarkerSize(2);
 //    h_resolution1->SetMarkerStyle(8);

    //h_resolution2->SetTitle(" ");
 //    h_resolution2->GetXaxis()->SetTitle("Beam energy (GeV)");
 //    h_resolution2->GetYaxis()->SetTitleOffset(1.4);
 //    h_resolution2->GetYaxis()->SetTitle("sigma/Mean (GeV)");
 //    h_resolution2->SetMarkerColorAlpha(kBlue+4, 0.95);
 //    h_resolution2->SetMarkerSize(2);
 //    h_resolution2->SetMarkerStyle(8);

   h_response->SetTitle(" ");
 //  h_response->GetXaxis()->SetTitle("Beam energy (GeV)");
 //  h_response->GetYaxis()->SetTitleOffset(1.4);
 //  h_response->GetYaxis()->SetTitle("Mean (GeV)/ Beam energy");
 //  h_response->SetMarkerColorAlpha(kPink+2, 0.95);
 //  h_response->SetMarkerSize(2);
 //  h_response->SetMarkerStyle(8);

  h_response1->SetTitle(" ");
 //  h_response1->GetXaxis()->SetTitle("Beam energy (GeV)");
 //  h_response1->GetYaxis()->SetTitleOffset(1.4);
 //  h_response1->GetYaxis()->SetTitle("Mean (GeV)/ Beam energy");
 //  h_response1->SetMarkerColorAlpha(kGreen+3, 0.95);
 //  h_response1->SetMarkerSize(2);
 //  h_response1->SetMarkerStyle(8);

 // h_response2->SetTitle(" ");
 //  h_response2->GetXaxis()->SetTitle("Beam energy (GeV)");
 //  h_response2->GetYaxis()->SetTitleOffset(1.4);
 //  h_response2->GetYaxis()->SetTitle("Mean (GeV)/ Beam energy");
 //  h_response2->SetMarkerColorAlpha(kBlue+4, 0.95);
 //  h_response2->SetMarkerSize(2);
 //  h_response2->SetMarkerStyle(8);

// TMultiGraph* mg = new TMultiGraph();
//   TMultiGraph* mg1 = new TMultiGraph();
  TCanvas *canvas_n1 = new TCanvas(hist_name, hist_name,600,600,1200,1200);
  canvas_n1->Range(-60.25,-0.625,562.25,0.625);
  canvas_n1->SetFillColor(0);
  canvas_n1->SetBorderMode(0);
  canvas_n1->SetBorderSize(2);
  // mg->SetTitle("resolution for pions  ");
  // mg->GetXaxis()->SetTitle("Beam energy (GeV)");
  // mg->GetYaxis()->SetTitleOffset(1.4);
  // mg->GetYaxis()->SetTitle("sigma/mean");
TAxis *axis5=  mg->GetYaxis();
  axis5->SetRangeUser(0.02,0.3);
  canvas_n1->SetGrid();
  auto legend = new TLegend(0.6,0.6,0.9,0.9);
  legend->SetHeader("","C");
   legend->AddEntry(h_resolution_shubham,"Previous studies","p");
  legend->AddEntry(h_resolution,"Validation","p");
  legend->AddEntry(h_resolution1,"Total_Tbdata","p");
  legend->AddEntry(h_resolution_EE,"SSinEE_Tbdata","p");
  legend->AddEntry(h_resolution_FH,"MipsinEE_Tbdata","p");
  //  legend->AddEntry(h_resolution_shubham,"previous studies","p");
  canvas_n1->cd();
  
  // mg->Add(h_resolution);
  // mg->Add(h_resolution2);
  gPad->Modified();
  gPad->Update();
  // h_resolution1->SetMinimum(0.02);
  // h_resolution1->SetMaximum(0.3);
  //h_resolution1->Draw("ALP");
 mg->SetTitle("");

  mg->Draw("ALP");
legend->Draw("sames");
gPad->Modified();
  gPad->Update();
  mg->SetMinimum(0.02);
  mg->SetMaximum(0.3);
 legend->Draw("sames");
  canvas_n1->Modified();
  canvas_n1->cd();
  canvas_n1->SetSelected(canvas_n1);
  canvas_n1->SaveAs(full_path1);

    TCanvas *canvas_n2 = new TCanvas(hist_name1, hist_name1,600,600,1200,1200);
  canvas_n2->Range(-60.25,-0.625,562.25,0.625);
  canvas_n2->SetFillColor(0);
  canvas_n2->SetBorderMode(0);
  canvas_n2->SetBorderSize(2);
  // mg1->SetTitle("response for pions  ");
  // mg1->GetXaxis()->SetTitle("Beam energy (GeV)");
  // mg1->GetYaxis()->SetTitleOffset(1.4);
  // mg1->GetYaxis()->SetTitle("mean/beam energy");
  // TAxis *axis51=  mg1->GetYaxis();
  // axis51->SetRangeUser(0.9,1.4);
  canvas_n2->SetGrid();
  //  mg1->GetHistogram()->GetYaxis()->SetRangeUser(0.5,1.3);
  // mg1->Add(h_response);
  // mg1->Add(h_response2);

  // mg1->Add
  gPad->Modified();
  gPad->Update();
 mg1->GetHistogram()->GetYaxis()->SetRangeUser(0.8,1.2);
 //h_response1->Draw("ALP");
  //  auto legend1 = new TLegend(0.75,0.75,0.9,0.9);
  // legend1->SetHeader("","C");
  // legend1->AddEntry(h_response,"Total","p");
  // legend1->AddEntry(h_response1,"SSinEE","p");
  // legend1->AddEntry(h_response2,"MipsinEE","p");
auto legend1 = new TLegend(0.6,0.1,0.9,0.4);
  legend1->SetHeader("","C");
  legend1->AddEntry(h_resolution,"Validation","p");
  legend1->AddEntry(h_resolution1,"Total_Tbdata","p");
  legend1->AddEntry(h_resolution_EE,"SSinEE_Tbdata","p");
  legend1->AddEntry(h_resolution_FH,"MipsinEE_Tbdata","p");

  mg1->SetTitle("");
  mg1->Draw("ALP");
  TLine* l=new TLine(0,1,350,1);
  l->Draw("sames");
  legend->Draw("sames");
  gPad->Modified();
  gPad->Update();

  canvas_n2->Modified();
  canvas_n2->cd();
  canvas_n2->SetSelected(canvas_n2);
  canvas_n2->SaveAs(full_path2);


  //  }
}
