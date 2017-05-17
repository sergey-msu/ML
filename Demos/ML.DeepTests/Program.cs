using System;
using System.Linq;

namespace ML.DeepTests
{
  class Program
  {
    static void Main(string[] args)
    {
      test();


      var runner =
        new OriginalMNIST();
        //new OriginalCIFAR10();
        //new OriginalCIFAR10Trunc();
        //new KaggleMNIST();
        //new KaggleCIFAR10();
        //new KaggleCatDog();
        //new KaggleCatDogTrunk();
        //new KaggleCatDog64Trunk();

      try
      {
        Console.WriteLine("STARTED at {0}", DateTime.Now);
        runner.Run();
        Console.WriteLine("DONE at {0}", DateTime.Now);
      }
      catch (Exception ex)
      {
        Console.WriteLine("ERROR:");
        Console.WriteLine(ex);

        Utils.SaveAlgCrushResults(runner.Alg, runner.OutputPath);
      }

      Console.ReadLine();
    }

    private static void test()
    {
      var data = new double[1][,] { new double[32,32] };
      var ker = new double[1][,] { new double[9,9] };

      using (var cat    = new System.Drawing.Bitmap(@"C:\Users\User\Desktop\a.bmp"))
      using (var kernel = new System.Drawing.Bitmap(@"C:\Users\User\Desktop\b.bmp"))
      {
        for (int x=0; x<32; x++)
        for (int y=0; y<32; y++)
          data[0][y, x] = (255.0D-cat.GetPixel(x, y).B)/255.0D;

        for (int x=0; x<9; x++)
        for (int y=0; y<9; y++)
          ker[0][y, x] = (255.0D-kernel.GetPixel(x, y).B)/255.0D;
      }

      var convolution = new DeepMethods.Models.ConvLayer(1, 9, padding: 4);
      convolution.InputDepth = 1;
      convolution.InputHeight = 32;
      convolution.InputWidth = 32;
      convolution._Build();

      for (int y=0; y<9; y++)
      for (int x=0; x<9; x++)
        convolution.SetKernel(0, 0, y, x, ker[0][y, x]);

      var res = convolution.Calculate(data);
      var max = res[0].Cast<double>().Max();

      using (var fm = new System.Drawing.Bitmap(32, 32))
      {
        for (int x=0; x<32; x++)
        for (int y=0; y<32; y++)
        {
          var val = 255.0D*(1.0D - res[0][y, x]/max);
          if (val>128) val = 255;
          fm.SetPixel(x, y, System.Drawing.Color.FromArgb((int)val, (int)val, (int)val));
        }

        fm.Save(@"C:\Users\User\Desktop\d.bmp");
      }
    }
  }
}
