using namespace Cuore;
int avg_pulse_t_to_f() {
    for (int t = 0; t < 19; t++) {
        TFile *f = TFile::Open(Form("/nfs/cuore1/scratch/branca/cuoresw_shift_aug18/avg/ds3054/average_pulses_ds3054_tower%.3d.root",t+1));
        TDirectory *global = f->GetDirectory("Global");
        for (int ch = 0; ch < 52; ch++) {
            int channel = t*52 + ch;
            QAverageVector *av = global->Get(Form("AveragePulses@Average_ds3054_chan%.4d", channel+1));
            if (av == NULL) {
                continue;
            }
            // Create FFT
            // Note that in module nFreqs == noiseVector.Size() --> The number of samples
            double samplingFreq = 1000;
            int nSamples = av->Size();
            double duration = nSamples / samplingFreq;
            // Initialize the spsVector
            QVector spsVec(nSamples);
            spsVec.Initialize(0);
            double mean = av->GetMean(nSamples);
            QVector zeroMeanPulse = *av - mean;
            QRealComplexFFT transformer(nSamples);
            transformer.SetWindowType(QFFT::StrToWindowType("Hann"), 2);
            QVectorC transformedPulse(nSamples);
            
            // Note we could do a mean subtracted av...
            int err = transformer.TransformToFreq(zeroMeanPulse, transformedPulse);
            
            QVector realPart = transformedPulse.Re();
            QVector imagPart = transformedPulse.Im();
            spsVec[0] = mean*mean;
            for (int k = 1; k < nSamples; k++) {spsVec[k] = realPart[k]*realPart[k] + imagPart[k]*imagPart[k];}
            // At this point spsVec should be a NPS vector
            QVector freq(nSamples);
            freq.Initialize(0);
            for (int k = 0; k < freq.Size(); k++) {freq[k] = k/duration;}
            double adc2mv = 1;
            // Try to create a PS graph?
            TGraph *g2 = new TGraph(freq.Size(), freq.GetArray(), spsVec.GetArray());
            int nBin = spsVec.Size()/2 + 1;
            double res = samplingFreq/spsVec.Size();
            TGraph *g = new TGraph(nBin - 1);
            for (int k = 1; k < nBin; k++) { double y; double x = k*res; y = (spsVec[k] * (adc2mv*adc2mv) / (spsVec.Size()*spsVec.Size() * res))*2; g->SetPoint(k-1, x, y);}
            g->SetName(Form("APfft%d", channel+1));
            if (channel == 0) {
                TFile *fOut = new TFile("/nfs/cuore1/scratch/welliver/test/avg/ds3054/AP_fft_test.root", "recreate");
            }
            else {
                TFile *fOut = TFile::Open("/nfs/cuore1/scratch/welliver/test/avg/ds3054/AP_fft_test.root", "update");
            }
            fOut->cd();
            g->Write();
            fOut->Close();
        }
        f->Close();
    }
    return 0;
}

