#include <stdio.h>
#include<fstream>
#include <vector>
#include <string>
#include <map>
#include<iostream>
using namespace std;
void fittingResolut(string file_name)//_valid, string file_name_test, string file_name_data)                                                                       
{
  char* hname = new char[200];
  char* name = new char[200];

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
 sprintf(path2,"Results/Reso_fits");//,file_name.c_str());
  TFile * inputfile = new TFile(hname,"READ");
const char *data[5] = { "Valid", "Train",
                         "Tbdata","QGSP","FTFP" };
const char *Energy[8]={"20","50","80","100","120","200","250","300"};
 int Elist[8]={20,50,80,100,120,200,250,300};
 int marker[5]={8, 22,21, 29, 34};
 int col[5]={kRed,kBlue,kBlack,kMagenta,kCyan+2};//,kBlue,kRed,kBlue+2,kMagenta,kCyan};
 TMultiGraph* mg = new TMultiGraph();
  TMultiGraph* mg1 = new TMultiGraph();
auto legend = new TLegend(0.55,0.6,0.9,0.9);
  legend->SetHeader("","C");
  gStyle->SetLegendTextSize(0.05);
  auto legend1 = new TLegend(0.55,0.6,0.9,0.9);
  legend1->SetHeader("","C");
  const char *eve_cat[3] = {"", "_SSinEE", "_MipsInEE"};
  const char *data1[3]={"Inclusive","SSinEE","MipsInEE"};
 // sprintf(full_path1,"%s/%s_overlay_resolution_v1.png",eve_cat[0],path2);
 //      sprintf(full_path2,"%s/%s_overlay_response_v1.png",path2);
      //const char *eve_cat[2] = { "SSinEE", "MipsInEE"};
  // for(int i=0;i<2;i++)
    
  //       {
  //  int marker[5]={8, 22,21, 29, 34};
  //int col[5]={kRed,kBlue,kBlack,kMagenta,kCyan+2};
 for (int i_data=0; i_data<2;i_data++)
    {
      sprintf(full_path1,"%s/%s_resolution_4GeV.png",path2,data[i_data]);
      //sprintf(full_path2,"%s/%s_response.png",path2,data[i_data]);
      TMultiGraph* mg = new TMultiGraph();
      TMultiGraph* mg1 = new TMultiGraph();
      auto legend = new TLegend(0.55,0.6,0.9,0.7);
      legend->SetHeader("","C");
      gStyle->SetLegendTextSize(0.05);
      auto legend1 = new TLegend(0.55,0.6,0.9,0.9);
      legend1->SetHeader("","C");

      // for (int i=0; i<3;i++)
      // 	{
      sprintf(hist_name1,"Resolution_%s",data[i_data]);
      //sprintf(hist_name,"Response%s_%s",eve_cat[i],data[i_data]);
	  //cout<<hist_name1<<endl;
	  //sprintf(full_path1,"%s/overlay_resolution.png",path2);
	  // sprintf(full_path2,"%s/overlay_response.png",path2);
      TGraphErrors* h_resolution_valid = (TGraphErrors*)inputfile->Get(hist_name1);
      //	  TGraphErrors* h_response_valid = (TGraphErrors*)inputfile->Get(hist_name);
	  h_resolution_valid->SetTitle(" ");
	  h_resolution_valid->GetXaxis()->SetTitle("Beam energy (GeV)");
	  h_resolution_valid->GetYaxis()->SetTitleOffset(1.4);
	  h_resolution_valid->GetYaxis()->SetTitle("Relative resolution(%)");
	  h_resolution_valid->SetMarkerColorAlpha(col[i_data], 0.95);
	  h_resolution_valid->SetMarkerSize(2);
          h_resolution_valid->SetMarkerStyle(marker[i_data]);
	  h_resolution_valid->SetLineColor(col[i_data]);
	  gStyle->SetOptFit(1);
	  TF1* fit_func = new TF1("fit_func","sqrt([0]*[0]+[1]*[1]/x+[2]*[2]/(x*x))",10,350);
	  fit_func->SetParameters(0.06,1.20,0.);
	  
	  fit_func->SetLineColor(kBlack);
	  fit_func->SetLineWidth(1);
	  fit_func->SetLineStyle(1);
	  h_resolution_valid->Fit(fit_func,"","R",10,350);        
	  cout<<endl<<fit_func->GetParameter(0)<<":"<<fit_func->GetParameter(1)<<":"<<fit_func->GetParameter(2)<<endl;	  
	  char* fit_params = new char[2000];
	  // sprintf(fit_params,"#splitline{S = (%0.1f #pm %0.1f)%% GeV^{0.5}}{C = (%0.1f #pm %0.1f)%%}",100*fit_func->GetParameter(1),100*fit_func->GetParError(1),100*fit_func->GetParameter(0),100*fit_func->GetParError(0));
	  sprintf(fit_params,"#splitline{S = (%0.1f #pm %0.1f)%%}{C = (%0.1f #pm %0.1f)%%}",100*fit_func->GetParameter(1),100*fit_func->GetParError(1),100*fit_func->GetParameter(0),100*fit_func->GetParError(0));
	  cout<<endl<<"S  = "<<100*fit_func->GetParameter(1)<<"+/-"<<100*fit_func->GetParError(1)
            << " , C = " <<100*fit_func->GetParameter(0)<<"+/-"<<100*fit_func->GetParError(0)<<endl;
	  
	  //h_resolution_valid->SetMarkerStyle(marker[i]);
	  // h_response_valid->SetTitle(" ");
	  // h_response_valid->GetXaxis()->SetTitle("Beam energy (GeV)");
	  // h_response_valid->GetYaxis()->SetTitleOffset(1.4);
	  // h_response_valid->GetYaxis()->SetTitle("Energy response");
	  // h_response_valid->SetMarkerColorAlpha(col[i], 0.95);
	  // h_response_valid->SetMarkerSize(2);
	  // //h_response_valid->SetMarkerStyle(marker[i]);
	  //mg->Add(h_resolution_valid);
	  // mg1->Add(h_response_valid);
	  legend->AddEntry(h_resolution_valid,"4GeVBins","p");
	  
	  //legend1->AddEntry(h_resolution_valid,data1[i],"p");
	
      TCanvas *canvas_n1 = new TCanvas(hist_name, hist_name,600,600,1200,1200);
      canvas_n1->Range(-60.25,-0.625,562.25,0.625);
      canvas_n1->SetFillColor(0);
      canvas_n1->SetBorderMode(0);
      canvas_n1->SetBorderSize(2);
      sprintf(name,"Resolution for %s",data[i_data]);
      h_resolution_valid->SetName(name);
      
    //   mg->SetTitle(name);
    //   mg->GetXaxis()->SetTitle("Beam energy (GeV)");
    //   mg->GetYaxis()->SetTitleOffset(1.4);
    //   mg->GetYaxis()->SetTitle("Relative resolution(%)");
    //   TAxis *axis5=  mg->GetYaxis();
    //   axis5->SetRangeUser(0.02,0.22);
      canvas_n1->SetGrid();
      canvas_n1->cd();
      gPad->Modified();
      gPad->Update();
      h_resolution_valid->Draw("AP");
      h_resolution_valid->SetMinimum(0.02);
      h_resolution_valid->SetMaximum(0.3 );
    //   mg->Draw("ALP");legend->Draw("sames");gPad->Modified();
      gPad->Update();
    //   mg->SetMinimum(0.02);
    //   mg->SetMaximum(0.3);
   legend->Draw("sames");
      canvas_n1->Modified();
      canvas_n1->cd();
      canvas_n1->SetSelected(canvas_n1);
      canvas_n1->SaveAs(full_path1);
    } 
    //   TCanvas *canvas_n2 = new TCanvas(hist_name1, hist_name1,600,600,1200,1200);
    //   canvas_n2->Range(-60.25,-0.625,562.25,0.625);
    //   canvas_n2->SetFillColor(0);
    //   canvas_n2->SetBorderMode(0);
    //   canvas_n2->SetBorderSize(2);
    //   sprintf(name,"Response for %s",data[i_data]);
    //   mg1->SetTitle(name);
    //   mg1->GetXaxis()->SetTitle("Beam energy (GeV)");
    //   mg1->GetYaxis()->SetTitleOffset(1.4);
    //   mg1->GetYaxis()->SetTitle("mean/beam energy");
    //   TAxis *axis51=  mg1->GetYaxis();
    //   axis51->SetRangeUser(0.9,1.4);
    //   canvas_n2->SetGrid();
    //   mg1->GetHistogram()->GetYaxis()->SetRangeUser(0.8,1.2);
    //   gPad->Modified();
    //   gPad->Update();
    //   mg1->Draw("ALP");
    //   TLine* l=new TLine(0,1,350,1);
    //   l->Draw("sames");
    //   legend1->Draw("sames");
    //   gPad->Modified();
    //   gPad->Update();
    //   canvas_n2->Modified();
    //   canvas_n2->cd();
    //   canvas_n2->SetSelected(canvas_n2);
    //   canvas_n2->SaveAs(full_path2);
    // }
 
}
