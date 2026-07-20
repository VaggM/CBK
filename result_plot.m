% Final EER% figure based on the saved models variable saved in models.mat
function result_plot(models, traindbname)

    % Model EER_{SF}(%)
    tot_fig = figure(1);

    tot_ax = axes('Parent', tot_fig);
    title(tot_ax, sprintf('ERR on All Models %s', traindbname));
    
    hold(tot_ax, 'on');
    grid(tot_ax, 'on');
    
    for i=1:length(models)
    
        error = zeros([length(models(i).k), 14]);
    
        for j=1:length(models(i).k)
    
            [n_writer, ~] = size(models(i).rs1.stats_test{1,1}(1).RESNREF);
            [n_iwriter, ~] = size(models(i).irs1.stats_test{1,1}(1).RESNREF);
    
            res1 = models(i).rs1.stats_test{1,j};
            ires1 = models(i).irs1.stats_test{1,j};
    
            res2 = models(i).rs2.stats_test{1,j};
            ires2 = models(i).irs2.stats_test{1,j};

            res3 = models(i).rs3.stats_test{1,j};
            ires3 = models(i).irs3.stats_test{1,j};

            res4 = models(i).rs4.stats_test{1,j};
            ires4 = models(i).irs4.stats_test{1,j};

            res5 = models(i).rs5.stats_test{1,j};
            ires5 = models(i).irs5.stats_test{1,j};
    
            for segment=1:14
    
                [err_mean_rs1, err_mean_rs2, err_mean_rs3, eer_mean_rs4, eer_mean_rs5] = deal(zeros(n_writer,1));
                [err_mean_irs1, err_mean_irs2, err_mean_irs3, eer_mean_irs4, eer_mean_irs5] = deal(zeros(n_iwriter,1));
    
                for writer=1:n_writer
                    err_sort_rs1 = sort([res1(segment).RESNREF(writer,:).SF1]);
                    err_sort_rs2 = sort([res2(segment).RESNREF(writer,:).SF1]);
                    err_sort_rs3 = sort([res3(segment).RESNREF(writer,:).SF1]);
                    err_sort_rs4 = sort([res4(segment).RESNREF(writer,:).SF1]);
                    err_sort_rs5 = sort([res5(segment).RESNREF(writer,:).SF1]);
    
                    err_mean_rs1(writer) = mean(err_sort_rs1(:));
                    err_mean_rs2(writer) = mean(err_sort_rs2(:));
                    err_mean_rs3(writer) = mean(err_sort_rs3(:));
                    err_mean_rs4(writer) = mean(err_sort_rs4(:));
                    err_mean_rs5(writer) = mean(err_sort_rs5(:));
                end
    
                for iwriter=1:n_iwriter
                    err_sort_irs1 = sort([ires1(segment).RESNREF(iwriter,:).SF1]);
                    err_sort_irs2 = sort([ires2(segment).RESNREF(iwriter,:).SF1]);
                    err_sort_irs3 = sort([ires3(segment).RESNREF(iwriter,:).SF1]);
                    err_sort_irs4 = sort([ires4(segment).RESNREF(iwriter,:).SF1]);
                    err_sort_irs5 = sort([ires5(segment).RESNREF(iwriter,:).SF1]);
    
                    err_mean_irs1(iwriter) = mean(err_sort_irs1);
                    err_mean_irs2(iwriter) = mean(err_sort_irs2);
                    err_mean_irs3(iwriter) = mean(err_sort_irs3);
                    err_mean_irs4(iwriter) = mean(err_sort_irs4);
                    err_mean_irs5(iwriter) = mean(err_sort_irs5);
                end
    
                error(j,segment) = mean([...
                    err_mean_rs1; err_mean_rs2; err_mean_rs3; eer_mean_rs4; eer_mean_rs5;...
                    err_mean_irs1; err_mean_irs2; err_mean_irs3; eer_mean_irs4; eer_mean_irs5...
                    ]);
                
            end
    
        end

    end    

end