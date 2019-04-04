%
% superclass for all individual signals
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-03-29
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
            % ensure samples is a cell array
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axes, samples );

            % check for row vectors
            for index_object = 1:numel( axes )

                % ensure row vectors
                if ~isrow( samples{ index_object } )
                    errorStruct.message     = sprintf( 'The content of samples{ %d } must be a row vector!', index_object );
                    errorStruct.identifier	= 'signal:NoRowVector';
                    error( errorStruct );
                end

            end % for index_object = 1:numel( axes )

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.signal_matrix( axes, samples );

        end % function objects = signal( axes, samples )

    end % methods

end % classdef signal < discretizations.signal_matrix
