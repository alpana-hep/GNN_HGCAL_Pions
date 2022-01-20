#include <stdio.h>
#include<fstream>
#include <vector>
#include <string>
#include <map>
#include<iostream>
using namespace std;
void DataOverlay(string file_name)//_valid, string file_name_test, string file_name_data)
{
  char* hname = new char[200];
  char* hist_name  = new char[200];
  char* hist_name1 = new char[200];
  char* hist_name2 = new char[200];
  char* hist_name3 = new char[200];
  char* hist_name4 = new char[200];
  char* hist_name5 = new char[200];
  char* hist_name6 = new char[200];
  char* hist_name7 = new char[200];
  char* hist_name8 = new char[200];
  char* hist_name9 = new char[200];
  char* title=new char[2000];
  char* full_path1 = new char[2000];
  char* full_path2 = new char[2000];
  char* full_path3 = new char[2000];
  char* full_path4 = new char[2000];
  char* path2 = new char[2000];

  char* hname1 = new char[2000];
  char* hname2 = new char[200];
  char* hname3 = new char[200];
  sprintf(hname,"%s",file_name.c_str());
  // sprintf(hname1,"%s",file_name_test.c_str());
  // sprintf(hname2,"%s",file_name_data.c_str());
  sprintf(path2,"Results/",file_name.c_str());
  TFile * inputfile = new TFile(hname,"READ");
  // TFile * inputfile1 = new TFile(hname1,"READ");
  // TFile * inputfile2 = new TFile(hname2,"READ");
  sprintf(hist_name,"Response_Valid");
  sprintf(hist_name1,"Resolution_Valid");
  //  sprintf(hist_name2,"resolution")
  sprintf(hist_name2,"Response_Train");
  sprintf(hist_name3,"Resolution_Train");
  sprintf(hist_name4,"Response_Tbdata");
  sprintf(hist_name5,"Resolution_Tbdata");
  
  sprintf(full_path1,"%s/overlay_resolution.png",path2);
  sprintf(full_path2,"%s/overlay_response.png",path2);
  // sprintf(full_path3,"%s/overlay_resolution_MipsInEE_v1.png",path2);
  // sprintf(full_path4,"%s/overlay_response_MipsInEE_v1.png",path2);


  TGraphErrors* h_resolution_valid = (TGraphErrors*)inputfile->Get(hist_name1);
  TGraphErrors* h_response_valid = (TGraphErrors*)inputfile->Get(hist_name);
  TGraphErrors* h_resolution_test = (TGraphErrors*)inputfile->Get(hist_name3);
  TGraphErrors* h_response_test = (TGraphErrors*)inputfile->Get(hist_name2);
 TGraphErrors* h_resolution_data = (TGraphErrors*)inputfile->Get(hist_name5);
  TGraphErrors* h_response_data = (TGraphErrors*)inputfile->Get(hist_name4);

  h_resolution_valid->SetTitle(" ");
  h_resolution_valid->GetXaxis()->SetTitle("Beam energy (GeV)");
  h_resolution_valid->GetYaxis()->SetTitleOffset(1.4);
  h_resolution_valid->GetYaxis()->SetTitle("sigma/Mean (GeV)");
  h_resolution_valid->SetMarkerColorAlpha(kRed, 0.95);
  h_resolution_valid->SetMarkerSize(2);
  h_resolution_valid->SetMarkerStyle(8);


  h_resolution_test->SetTitle(" ");
  h_resolution_test->GetXaxis()->SetTitle("Beam energy (GeV)");
  h_resolution_test->GetYaxis()->SetTitleOffset(1.4);
  h_resolution_test->GetYaxis()->SetTitle("sigma/Mean (GeV)");
  h_resolution_test->SetMarkerColorAlpha(kBlue, 0.95);
  h_resolution_test->SetMarkerSize(2);
  h_resolution_test->SetMarkerStyle(kFullTriangleUp);

  h_resolution_data->SetTitle(" ");
  h_resolution_data->GetXaxis()->SetTitle("Beam energy (GeV)");
  h_resolution_data->GetYaxis()->SetTitleOffset(1.4);
  h_resolution_data->GetYaxis()->SetTitle("sigma/Mean (GeV)");
  h_resolution_data->SetMarkerColorAlpha(kBlack, 0.95);
  h_resolution_data->SetMarkerSize(2);
  h_resolution_data->SetMarkerStyle(kFullSquare);

  h_response_valid->SetTitle(" ");
  h_response_valid->GetXaxis()->SetTitle("Beam energy (GeV)");
  h_response_valid->GetYaxis()->SetTitleOffset(1.4);
  h_response_valid->GetYaxis()->SetTitle("Mean (GeV)/ Beam energy");
  h_response_valid->SetMarkerColorAlpha(kRed, 0.95);
  h_response_valid->SetMarkerSize(2);
  h_response_valid->SetMarkerStyle(8);

  h_response_test->SetTitle(" ");
  h_response_test->GetXaxis()->SetTitle("Beam energy (GeV)");
  h_response_test->GetYaxis()->SetTitleOffset(1.4);
  h_response_test->GetYaxis()->SetTitle("Mean (GeV)/ Beam energy");
  h_response_test->SetMarkerColorAlpha(kBlue, 0.95);
  h_response_test->SetMarkerSize(2);
  h_response_test->SetMarkerStyle(kFullTriangleUp);

  h_response_data->SetTitle(" ");
  h_response_data->GetXaxis()->SetTitle("Beam energy (GeV)");
  h_response_data->GetYaxis()->SetTitleOffset(1.4);
  h_response_data->GetYaxis()->SetTitle("Mean (GeV)/ Beam energy");
  h_response_data->SetMarkerColorAlpha(kBlack, 0.95);
  h_response_data->SetMarkerSize(2);
  h_response_data->SetMarkerStyle(kFullSquare);
  TMultiGraph* mg = new TMultiGraph();
  TMultiGraph* mg1 = new TMultiGraph();

  TCanvas *canvas_n1 = new TCanvas(hist_name, hist_name,600,600,1200,1200);
  canvas_n1->Range(-60.25,-0.625,562.25,0.625);
  canvas_n1->SetFillColor(0);
  canvas_n1->SetBorderMode(0);
  canvas_n1->SetBorderSize(2);
  mg->SetTitle("resolution for pions  ");
  mg->GetXaxis()->SetTitle("Beam energy (GeV)");
  mg->GetYaxis()->SetTitleOffset(1.4);
  mg->GetYaxis()->SetTitle("sigma/mean");
  // mg->Add(h_resolution_valid);
  // mg->Add(h_resolution_data);
  // mg->Add(h_resolution_test);


  TAxis *axis5=  mg->GetYaxis();
  axis5->SetRangeUser(0.02,0.3);
  canvas_n1->SetGrid();
  auto legend = new TLegend(0.7,0.7,0.9,0.9);
  legend->SetHeader("","C");
  legend->AddEntry(h_resolution_valid,"valid","p");
  legend->AddEntry(h_resolution_test,"training","p");
  legend->AddEntry(h_resolution_data,"Data","p");
  gStyle->SetLegendTextSize(0.05);
  canvas_n1->cd();
  mg->Add(h_resolution_valid);
  mg->Add(h_resolution_data);
  //  mg->Add(h_resolution_test);
  gPad->Modified();
  gPad->Update();
  h_resolution_test->SetMinimum(0.02);
  h_resolution_test->SetMaximum(0.3);
  h_resolution_test->Draw("ALP");
  
  mg->Draw("LP");
   //   c3->BuildLegend();
  //  mg->Draw("ALP");
  legend->Draw("sames");
  //legend->Draw();
  gPad->Modified();
  gPad->Update();
  mg->SetMinimum(0.02);
  mg->SetMaximum(0.25);

  // h_resolution_valid->Draw("ALP"); 
  // h_resolution_data->Draw("ALP sames");
  // h_resolution_test->Draw("ALP sames");
  //  h_resolution_valid->Draw("ALP sames");
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
  mg1->SetTitle("response for pions  ");
  mg1->GetXaxis()->SetTitle("Beam energy (GeV)");
  mg1->GetYaxis()->SetTitleOffset(1.4);
  mg1->GetYaxis()->SetTitle("mean/beam energy");
  TAxis *axis51=  mg1->GetYaxis();
  axis51->SetRangeUser(0.9,1.4);
  canvas_n2->SetGrid();
  mg1->GetHistogram()->GetYaxis()->SetRangeUser(0.5,1.3);
  mg1->Add(h_response_valid);
   mg1->Add(h_response_data);
  gPad->Modified();
  gPad->Update();
  h_response_test->GetHistogram()->GetYaxis()->SetRangeUser(0.8,1.2);
  h_response_test->Draw("ALP");
   auto legend1 = new TLegend(0.7,0.1,0.9,0.3);
  legend1->SetHeader("","C");
  legend1->AddEntry(h_resolution_valid,"valid","p");
  legend1->AddEntry(h_resolution_test,"training","p");
 legend1->AddEntry(h_resolution_data,"Data","p");
 gStyle->SetLegendTextSize(0.05);
  mg1->Draw("LP");
  TLine* l=new TLine(0,1,350,1);
  l->Draw("sames");
  legend1->Draw("sames");
  gPad->Modified();
  gPad->Update();
  // mg1->SetMinimum(0.1);
  // mg1->SetMaximum(1.4);

  canvas_n2->Modified();
  canvas_n2->cd();
  canvas_n2->SetSelected(canvas_n2);
  canvas_n2->SaveAs(full_path2);

}
