using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML.Core;
using ML.Contracts;
using System.IO;

namespace ML.ConsoleTest
{
  public class Visualizer
  {
    private const string COMMA = ",";

    private int m_XIdx;
    private int m_YIdx;
    private double[] m_Shear;
    private int m_InputDim;

    private double m_XMin;
    private double m_XMax;
    private double m_YMin;
    private double m_YMax;
    private int   m_XCnt;
    private int   m_YCnt;

    public Visualizer(DataWrapper data)
    {
      Data = data;
    }

    public readonly DataWrapper Data;

    public void Run(ISupervisedAlgorithm algorithm, int xidx=0, int yidx=1, double[] shear = null)
    {
      m_XIdx = xidx;
      m_YIdx = yidx;
      m_Shear = shear;
      m_InputDim = algorithm.TrainingSample.First().Key.Length;

      var fn = string.Format("data/{0}-{1:yyyyMMdd-hhmmss}.csv", algorithm.ID, DateTime.Now);
      using (var file = File.Create(fn))
      using (var writer = new StreamWriter(file))
      {
        prepareCanvas();
        doCalc(algorithm, writer);
      }
    }

    private void prepareCanvas()
    {
      m_XMin = Data.Data.Min(d => d.Key[m_XIdx]);
      m_XMax = Data.Data.Max(d => d.Key[m_XIdx]);
      m_YMin = Data.Data.Min(d => d.Key[m_YIdx]);
      m_YMax = Data.Data.Max(d => d.Key[m_YIdx]);

      m_XCnt = 200;
      m_YCnt = 200;
    }

    private void doCalc(ISupervisedAlgorithm algorithm, StreamWriter writer)
    {
      for (var k=0; k<Math.Max(m_XCnt, m_YCnt); k++)
      {
        var px = (k<m_XCnt) ? getProjPoint(k, 0)[0].ToString() : string.Empty;
        var py = (k<m_YCnt) ? getProjPoint(0, k)[1].ToString() : string.Empty;
        var pd = string.Empty;
        if (k<m_YCnt)
        {
          var data = new double[m_XCnt];
          for (int i=0; i<m_XCnt; i++)
          {
            var point = getPoint(i, k);
            var cls = algorithm.Classify(point);
            data[i] = cls.Value;
          }
          pd = string.Join(COMMA, data);
        }

        writer.WriteLine("{0},{1},{2}", px, py, pd);
      }
    }

    private double[] getPoint(int i, int j)
    {
      var result = new double[m_InputDim];
      if (m_Shear != null)
      {
        for (int k=0; k<m_InputDim; k++)
          result[k] = m_Shear[k];
      }

      result[m_XIdx] = m_XMin + (m_XMax-m_XMin)*i/(m_XCnt-1);
      result[m_YIdx] = m_YMin + (m_YMax-m_YMin)*j/(m_YCnt-1);

      return result;
    }

    private double[] getProjPoint(int i, int j)
    {
      var result = new double[2];
      result[0] = m_XMin + (m_XMax-m_XMin)*i/(m_XCnt-1);
      result[1] = m_YMin + (m_YMax-m_YMin)*j/(m_YCnt-1);
      return result;
    }
  }
}
