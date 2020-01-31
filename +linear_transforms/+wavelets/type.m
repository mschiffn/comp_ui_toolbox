%
% abstract superclass for all wavelet types
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
%
classdef (Abstract) type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = type( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'type:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create regularization options
            %--------------------------------------------------------------
            % repeat default regularization options
            objects = repmat( objects, size );

        end % function objects = type( size )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        QMFs = MakeONFilter( types )

        %------------------------------------------------------------------
        % generate orthogonal boundary conjugate mirror filters
        % (Cohen-Daubechies-Jawerth-Vial)
        %------------------------------------------------------------------
%         OBCMFs = MakeOBFilter( types )

        %------------------------------------------------------------------
        % set up filters for CDJV Wavelet Transform
        %------------------------------------------------------------------
%         QMFs = MakeCDJVFilter( types )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        strs_out = string( types )

	end % methods (Abstract)

end % classdef (Abstract) type
