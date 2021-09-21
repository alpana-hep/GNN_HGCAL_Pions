#include <stdio.h>
#include<fstream>
#include <vector>
#include <string>
#include <map>
#include<iostream>
using namespace std;
double calculateError(float x, float y, float ex, float ey,float cov )
{
  return (x/y)*(sqrt(((ex/x)*(ex/x))+((ey/y)*(ey/y))));//-(2*cov/x*y)));
    }
void plotResolution(string filename)
{
  char* hname = new char[200];
 char* hist_name  = new char[200];
  char* hist_name1 = new char[200];
  char* hist_name2 = new char[200];
char* full_path = new char[2000];
  char* full_path1 = new char[2000];
  char* full_path2 = new char[2000];
  char* path2 = new char[2000];
  char* title= new char[2000];

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
  sprintf(path2,"Results",filename.c_str());
  TFile * inputfile = new TFile(hname,"READ");
  // sprintf(full_path,"%s/h_predbE.png",path2);
  // sprintf(full_path1,"%s/h_true_EnbE.png",path2);
  // sprintf(full_path2,"%s/h_fracbE.png",path2);
  // sprintf(full_path3,"%s/h_normbE.png",path2);
  // sprintf(full_path4,"%s/h_frac_norm_20bE.png",path2);
  char* reso_name=new char[1000];
  char* resp_name=new char[1000];
  char* name=new char[1000];
  int rebin=2;
  TFile* file = new TFile("PionResolution_flaten_nosemi_78epoch_1M_scaled.root","RECREATE");
 const char *data[5] = { "Valid", "Train",
			 "Tbdata","QGSP","FTFP" };
   // int Elist[85] =  {9, 14 , 18 , 22 , 26 , 30,  34 , 38 , 42,  46,  50,  54,  58,  62,  66,  70,  74,  78,
   // 		     82,  86,  90,  94,  98, 102, 106, 110, 114 ,118, 122, 126, 130, 134, 138, 142 ,146, 150,
   // 		     154 ,158 ,162 ,166 ,170 ,174 ,178, 182, 186, 190 ,194 ,198, 202, 206, 210, 214 ,218, 222,
   // 		     226, 230, 234 ,238, 242 ,246 ,250, 254, 258 ,262 ,266, 270, 274, 278, 282, 286, 290, 294,
   // 		     298, 302 ,306, 310, 314, 318 ,322, 326, 330, 334 ,338 ,342 ,346};
 const char *Energy[8]={"20","50","80","100","120","200","250","300"};
 int Elist[8]={20,50,80,100,120,200,250,300};
  for (int i_data=0; i_data<5;i_data++)
    {
      sprintf(reso_name,"Resolution_%s",data[i_data]);
      sprintf(resp_name,"Response_%s",data[i_data]);
      TGraphErrors* h_resolution = new TGraphErrors();
      TGraphErrors* h_response = new TGraphErrors();
      h_response->SetTitle("response for pions ");
      h_response->GetXaxis()->SetTitle("Beam energy (GeV)");
      h_response->GetYaxis()->SetTitleOffset(1.4);
      h_response->GetYaxis()->SetTitle("Mean (GeV)/True beam");
      h_response->SetMarkerColorAlpha(kBlue, 0.95);
      h_response->SetMarkerSize(2);
      h_response->SetMarkerStyle(8);

      h_resolution->SetTitle("resolution for pions  ");
      h_resolution->GetXaxis()->SetTitle("Beam energy (GeV)");
      h_resolution->GetYaxis()->SetTitleOffset(1.4);
      h_resolution->GetYaxis()->SetTitle("sigma/mean");

      h_resolution->SetMarkerColorAlpha(kRed, 0.95);
      h_resolution->SetMarkerSize(2);
      h_resolution->SetMarkerStyle(8);
      h_response->SetName(resp_name);
      h_resolution->SetName(reso_name);

      //
      
      //      h_resolution->SetLineColor(kRed);
      // h_resolution->GetAttLine(0)->SetLineColor(kRed);
      // h_resolution->GetAttLine(1)->SetLineColor(kBlue);
      float xmin=0, xmax=0,xhigh=0;
      int inext=0;
      float mean=0, sigma=0, mean_err=0,sigma_err=0,Err_resolution=0, Err_response=0,resolution=0, response=0;
      //      TF1* fit_func_data = new TF1("fit_func_data","gaus");
      for(int i_en=0;i_en<8;i_en++)
	{
	  if(Elist[i_en]<=80)	
	    xhigh= 2.5*Elist[i_en];
	  else
	    xhigh= 2.0*Elist[i_en];
	  //	  cout<<xhigh<<endl;
	  //TF1* fit_func_data = new TF1("fit_func_data","gaus");

	  sprintf(hist_name,"%s_Predi_TrueEn_%d",data[i_data],Elist[i_en]);//,Elist[i_en]+4);
	  TH1F* resp= (TH1F*)inputfile->Get(hist_name);
	  TCanvas *Canvas_n2 = new TCanvas(hist_name, hist_name,600,600,1200,1200);
          Canvas_n2->Range(-60.25,-0.625,562.25,0.625);
          Canvas_n2->SetFillColor(0);
          Canvas_n2->SetBorderMode(0);
          Canvas_n2->SetBorderSize(2);
	  TPaveStats *ptstats2 = new TPaveStats(0.6,0.6,0.9,0.9,"brNDC");
	  ptstats2->SetBorderSize(1);
	  ptstats2->SetFillColor(0);
	  ptstats2->SetLineColor(kRed);
	  ptstats2->SetTextAlign(12);
	  ptstats2->SetTextColor(kRed);
	  ptstats2->SetTextFont(42);
	  ptstats2->SetOptStat(1111);
	  ptstats2->SetOptFit(1);
	  resp->GetListOfFunctions()->Add(ptstats2);
	  ptstats2->SetParent(resp);
	  resp->SetLineWidth(2);
	  sprintf(full_path,"%s/%s_PredEn_%03d.png",path2,data[i_data],Elist[i_en]);
          resp->GetXaxis()->SetTitle("Predicted energy in GeV");
	  sprintf(name,"PredEn_%d",Elist[i_en]);
	  sprintf(title,"PredEn (%d GeV) for %s dataset",Elist[i_en],data[i_data]);
          resp->SetName(name);                                                                                                                                                             
          resp->SetTitle(title);
	  if(i_en==0)
	    {
	  xmin=resp->GetMean()-1.5*resp->GetRMS();
	  xmax=resp->GetMean()+ 1.0*resp->GetRMS();

	    }
	  else
	    {
          xmin=resp->GetMean()-1.4*resp->GetRMS();
          xmax=resp->GetMean()+ 1.2*resp->GetRMS();
	    }
	  resp->Rebin(rebin);  
	  TF1* fit_func = new TF1("fit_func","gaus",20,300);

	  //fit_func_data->SetParameters(10000.0,resp->GetMean(),resp->GetRMS());
	  fit_func->SetRange(xmin,xmax);
          gStyle->SetOptFit(1);
          resp->Fit("fit_func","MR");//,"",xmin,xmax);
	  // TF1 *fit_func_data2 = resp->GetFunction("fit_func_data");
	  // xmin=fit_func_data2->GetParameter(1)-1.5*fit_func_data2->GetParameter(2);
	  // xmax = fit_func_data2->GetParameter(1)+1.0*fit_func_data2->GetParameter(2);
	  // fit_func_data2->SetRange(xmin,xmax);

	  // resp->Fit("fit_func_data","R","",xmin,xmax);
	  // TF1 *fit_func_data3 = resp->GetFunction("fit_func_data");
          // xmin=fit_func_data3->GetParameter(1)-1.5*fit_func_data3->GetParameter(2);
          // xmax = fit_func_data3->GetParameter(1)+1.0*fit_func_data3->GetParameter(2);
	  // fit_func_data3->SetRange(xmin,xmax);
	  //resp->Fit("fit_func_data","R","",xmin,xmax);

	  // resp->Rebin(rebin);
	  resp->Draw();
	  // TFitResultPtr r = resp->Fit("fit_func_data", "RS");  

	  // // Get all information from the fit
	  // TMatrixDSym cov = r->GetCovarianceMatrix();
	  //	  cout<<cov[1][2]<<endl;
          resp->GetFunction("fit_func")->SetLineColor(kRed);
          TF1* f = resp->GetFunction("fit_func");
	  //f->Draw("sames");
          mean= f->GetParameter(1);
          sigma=f->GetParameter(2);
	  //	  cout<<"corre"<<"\t"<<cov<<endl;
	  mean_err= f->GetParError(1);
	  sigma_err=f->GetParError(2);
	  resolution = sigma/mean;
	  // if(i_en==0)
	  //   inext=3;
	  // else
	  //   inext=2;
          response=  mean/Elist[i_en];//+inext);
	  Err_resolution = calculateError(sigma,mean,sigma_err,mean_err,0.0);
	  cout<<Err_resolution<<"\error"<<endl;
	  Err_response = mean_err/Elist[i_en];
	  h_resolution->SetPoint(i_en,Elist[i_en],resolution);
          h_response->SetPoint(i_en,Elist[i_en],response);
	  h_resolution->SetPointError(i_en,0,Err_resolution);
          h_response->SetPointError(i_en,0,Err_response);
	  resp->GetXaxis()->SetRangeUser(0,xhigh);
	  //	  resp->Draw();
	  Canvas_n2->Modified();
          Canvas_n2->cd();
          Canvas_n2->SetSelected(Canvas_n2);
          Canvas_n2->SaveAs(full_path);
	}
      TCanvas *Canvas_n2 = new TCanvas(hist_name, hist_name,800,800,1400,1400);
      Canvas_n2->Range(-60.25,-0.625,562.25,0.625);
      Canvas_n2->SetFillColor(0);
      Canvas_n2->SetBorderMode(0);
      Canvas_n2->SetBorderSize(2);
      
      Canvas_n2->SetGrid();
      //      h_resolution->GetXaxis()->SetRangeUser(-0.6,0);
      h_resolution->Draw("ALP");
      Canvas_n2->Modified();
      Canvas_n2->cd();
      sprintf(full_path2,"%s/Resolution_%s.png",path2,data[i_data]);

      Canvas_n2->SetSelected(Canvas_n2);
      Canvas_n2->SaveAs(full_path2);
      TCanvas *Canvas_n1 = new TCanvas(hist_name1, hist_name1,800,800,1400,1400);
      Canvas_n1->Range(-60.25,-0.625,562.25,0.625);
      Canvas_n1->SetFillColor(0);
      Canvas_n1->SetBorderMode(0);
      Canvas_n1->SetBorderSize(2);

      Canvas_n1->SetGrid();
      TLine* l=new TLine(0,1,300,1);
      h_response->Draw("ALP");


      l->Draw("sames");
      sprintf(full_path,"%s/Response_%s.png",path2,data[i_data]);
      Canvas_n1->Modified();
      Canvas_n1->cd();
      Canvas_n1->SetSelected(Canvas_n1);
      Canvas_n1->SaveAs(full_path);

      file->cd();
      h_response->Write();
      h_resolution->Write();
      //file->Close();

    }
  file->Close();
}
