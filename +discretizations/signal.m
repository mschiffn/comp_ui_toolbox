%
% superclass for all individual signals
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-05-04
%
classdef signal < discretizations.signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
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

            % ensure row vectors
            indicator_row = cellfun( @isrow, samples );
            if ~all( indicator_row( : ) )
                errorStruct.message = 'samples must be row vectors!';
                errorStruct.identifier = 'signal:NoRowVectors';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.signal_matrix( axes, samples );

        end % function objects = signal( axes, samples )

    end % methods

end % classdef signal < discretizations.signal_matrix
