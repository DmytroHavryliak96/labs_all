using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Npgsql;
using NeuralNetwork;

namespace Labs.LabsFolder.Lab2
{
    public partial class Form2 : Form
    {
        private DataTable dt = new DataTable();
        private BackPropagationNetwork bpn;
        private int[] layerSizes;
        private double[][] input;
        private double[][] desired;
        private TransferFunction[] TFunc;
        private double TrainingRate = 0.15;
        private double Momentum = 0.1;
        private double error;
        private DataTable errors = new DataTable();
        private DataTable function = new DataTable();
        private DataTable weights = new DataTable();

        public Form2()
        {
            InitializeComponent();
        }

        private void завантажитиToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                string connstring = "Server=127.0.0.1;Port=5432;User Id=postgres;Password=1postgres;Database=Labs;";
                NpgsqlConnection conn = new NpgsqlConnection(connstring);
                conn.Open();
                string sql = "SELECT * FROM lab2";
                NpgsqlDataAdapter da = new NpgsqlDataAdapter(sql, conn);
                da.Fill(dt);
                input = new double[dt.Rows.Count][];
                desired = new double[dt.Rows.Count][];

                for (int i =0; i<dt.Rows.Count; i++) 
                {
                    input[i] = new double[dt.Columns.Count-1];
                }


                for (int i = 0; i < dt.Rows.Count; i++)
                    for (int j = 1; j < dt.Columns.Count; j++)
                        input[i][j-1] = Convert.ToDouble(dt.Rows[i][j]);

                conn.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }

        }

        private void створитиМережуToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormCreate f3 = new FormCreate();
            this.Hide();
            f3.ShowDialog();
            layerSizes = new int[f3.layerSizes.Length];
            for(int i =0; i<f3.layerSizes.Length; i++)
            {
                layerSizes[i] = f3.layerSizes[i];
            }
           
            TFunc = new TransferFunction[f3.TFunc.Length];
            for (int i = 0; i < f3.TFunc.Length; i++)
            {
                TFunc[i] = f3.TFunc[i];
            }

            f3.Close();

            for (int i = 0; i < dt.Rows.Count; i++)
            {
                desired[i] = new double[layerSizes.Last()];
            }
           

            bpn = new BackPropagationNetwork(layerSizes, TFunc);
            this.Show();
        }

        private void навчитиToolStripMenuItem_Click(object sender, EventArgs e)
        {
            dataGridView1.DataSource = errors;
            dataGridView2.DataSource = function;

            switch (comboBox1.Text)
            {
                case "ln|cosx1| + tgx2 + ctgx3":
                    {
                        errors.Columns.Add("iteration");
                        errors.Columns.Add("error");
                        errors.Columns.Add("output[d1]");
                        errors.Columns.Add("output[d2]");
                        double sum = 0.0;
                        for(int i = 0; i< dt.Rows.Count; i++)
                        {
                            desired[i][0] = Math.Log(Math.Abs(Math.Cos(input[i][0]))) + Math.Tan(input[i][1]) + 1 / Math.Tan(input[i][2]);
                            sum += desired[i][0];
                        }

                        double average = sum / (desired.GetUpperBound(0) + 1);
                        for (int i = 0; i < dt.Rows.Count; i++)
                        {
                            if (desired[i][0] > average)
                                desired[i][1] = 1;
                            else
                                desired[i][1] = 0;
                        }

                        function.Columns.Add("i");
                        function.Columns.Add("d1");
                        function.Columns.Add("d2");
                        function.Columns.Add("average");

                        for (int i = 0; i< desired.GetUpperBound(0)+1; i++)
                        {
                            DataRow row = function.NewRow();
                            row["i"] = i;
                            row["d1"] = desired[i][0];
                            row["d2"] = desired[i][1];
                            row["average"] = average;
                            function.Rows.Add(row);
                        }
                        double[] output = new double[2];

                        // Навчання
                        for (int i = 0; i < 1000; i++)
                        {
                            // Перемішуємо елементи вхідних даних для навчання
                            /*Random rnd = new Random();
                            for (int k = 0; k < input.GetUpperBound(0)+1; k++)
                            {
                                int temp = rnd.Next(k, input.GetUpperBound(0) + 1);
                                double[] value;
                                value = input[k];
                                input[k] = input[temp];
                                input[temp] = value;
                                 
                            }*/

                           // for (int j = 0; j < dt.Rows.Count; j++)
                                error = bpn.Train(ref input[0], ref desired[0], TrainingRate, Momentum);
                            if (i % 100 == 0)
                            {
                                bpn.Run(ref input[0], out output);
                                DataRow row = errors.NewRow();
                                row["iteration"] = i;
                                row["error"] = error;
                                row["output[d1]"] = output[0];
                                row["output[d2]"] = output[1];
                                errors.Rows.Add(row);
                            }


                        }
                            break;
                    }

            }
        }
    }
}
