%
% abstract superclass for all wavelet types
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-04-16
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
            % 2.) create wavelet types
            %--------------------------------------------------------------
            % repeat default wavelet types
            objects = repmat( objects, size );

        end % function objects = type( size )

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( types )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.type
            if ~isa( types, 'linear_transforms.wavelets.type' )
                errorStruct.message = 'types must be linear_transforms.wavelets.type!';
                errorStruct.identifier = 'MakeONFilter:NoWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( types ) );

            % iterate wavelet types
            for index_object = 1:numel( types )

                % generate orthonormal QMF (scalar)
                QMFs{ index_object } = MakeONFilter_scalar( types( index_object ) );

            end % for index_object = 1:numel( types )

            % avoid cell array for single types
            if isscalar( types )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( types )

        %------------------------------------------------------------------
        % generate orthogonal boundary conjugate mirror filters
        % (Cohen-Daubechies-Jawerth-Vial)
        %------------------------------------------------------------------
%         OBCMFs = MakeOBFilter( types )

        %------------------------------------------------------------------
        % set up filters for CDJV Wavelet Transform
        %------------------------------------------------------------------
%         QMFs = MakeCDJVFilter( types )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        strs_out = string( types )

	end % methods (Abstract)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % generate orthonormal QMF (scalar)
        %------------------------------------------------------------------
        QMFs = MakeONFilter_scalar( types )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) type
