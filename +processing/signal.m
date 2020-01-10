%
% superclass for all individual signals
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2020-01-10
%
classdef signal < processing.signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal( axes, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure column vectors
            indicator_row = cellfun( @( x ) ~iscolumn( x ), samples );
            if any( indicator_row( : ) )
                errorStruct.message = 'samples must be column vectors!';
                errorStruct.identifier = 'signal:NoColumnVectors';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@processing.signal_matrix( axes, samples );

        end % function objects = signal( axes, samples )

    end % methods

end % classdef signal < processing.signal_matrix
