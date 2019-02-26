%
% superclass for all discrete time sets
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-02-20
%
classdef set_discrete_time < discretizations.set_discrete_physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_time( input )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                input = physical_values.time( 0 );
            end

            % ensure cell array
            if ~iscell( input )
                input = { input };
            end

            % ensure class physical_values.time
            for index_cell = 1:numel( input )
                if ~isa( input{ index_cell }, 'physical_values.time' )
                    errorStruct.message     = sprintf( 'input{ %d } must be physical_values.time!', index_cell );
                    errorStruct.identifier	= 'set_discrete_time:NoTime';
                    error( errorStruct );
                end
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.set_discrete_physical_value( input );

        end % function objects = set_discrete_time( input )

	end % methods

end % classdef set_discrete_time
