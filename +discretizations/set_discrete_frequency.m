%
% superclass for all discrete frequency sets
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-03-11
%
classdef set_discrete_frequency < discretizations.set_discrete_physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_frequency( input )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                input = physical_values.frequency( 1 );
            end

            % ensure cell array
            if ~iscell( input )
                input = { input };
            end

            % ensure class physical_values.frequency
            for index_cell = 1:numel( input )
                if ~isa( input{ index_cell }, 'physical_values.frequency' )
                    errorStruct.message     = sprintf( 'input{ %d } must be physical_values.frequency!', index_cell );
                    errorStruct.identifier	= 'set_discrete_time:NoFrequency';
                    error( errorStruct );
                end
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.set_discrete_physical_value( input );

        end % function objects = set_discrete_frequency( input )

        %------------------------------------------------------------------
        % union
        %------------------------------------------------------------------
        function [ set_out, ia, ic ] = union( sets_in )

            % retain unique discrete frequencies
            [ S, ia, ic ] = unique( [ sets_in.S ] );
            set_out = discretizations.set_discrete_frequency( S );

        end % function [ set_out, ia, ic ] = union( sets_in )

	end % methods

end % classdef set_discrete_frequency < discretizations.set_discrete_physical_value
