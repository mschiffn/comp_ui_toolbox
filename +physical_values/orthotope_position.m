%
% superclass for all position orthotopes
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-03-25
%
classdef orthotope_position < physical_values.orthotope

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthotope_position( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class physical_values.interval_length for first argument (additional arguments must match in superclass)
            if ~isa( varargin{ 1 }, 'physical_values.interval_length' )
                errorStruct.message     = sprintf( 'Argument %d must be physical_values.interval_length!', 1 );
                errorStruct.identifier	= 'orthotope_position:NoPositionInterval';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.orthotope( varargin{ : } );

        end

    end

end % classdef orthotope_position
